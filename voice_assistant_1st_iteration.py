# ==========================================================
# SAMWISE VOICE ASSISTANT
# ==========================================================

import queue
import threading
import tempfile
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz

from groq import Groq
from google import genai

from datetime import datetime

import torch
from silero_vad import load_silero_vad, VADIterator
# ==========================================================
# API KEYS
# ==========================================================

GROQ_API_KEY = "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"

GEMINI_API_KEY = "AQ.Ab8RN6LfH2VURtpZiPbSUCxSoJg5DcF9hzLCm5tZ-Se6Iwimig"

# ==========================================================
# SETTINGS
# ==========================================================

MIC_DEVICE = 1

SAMPLE_RATE = 16000
CHANNELS = 1

BLOCK_SIZE = 512

# Audio recording
PRE_BUFFER_SECONDS = 0.5

# IMPORTANT:
# How long silence must continue AFTER speech before
# we finalize the utterance.
POST_BUFFER_SECONDS = 1.7

# Maximum length of one utterance
MAX_RECORDING_SECONDS = 15

# Silero VAD
VAD_THRESHOLD = 0.40
MIN_SILENCE_DURATION_MS = 800

# Conversation
WAKE_TIMEOUT = 15.0

WAKE_VARIANTS = [
    "sam",
    "ssam",
    "samh",
    "saam",
    "samm",
    "som"
]
# ==========================================================
# AI CLIENTS
# ==========================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)
# ==========================================================
# CONVERSATION MEMORY
# ==========================================================

conversation = [
    {
        "role": "system",
        "content": """
You are Samwise.

You are an intelligent English voice assistant for a smart home automation system.

Rules:

- Keep replies concise.
- Be friendly and natural.
- Never mention AI models.
- Answer general questions normally.
- For smart home commands respond ONLY like:

[HOME]
TURN_ON BEDROOM_LIGHT

or

[HOME]
TURN_OFF FAN

No explanation with HOME commands.
"""
    }
]
# ==========================================================
# CONVERSATION MODE
# ==========================================================

conversation_active = False
last_command_time = 0.0
# ==========================================================
# LOAD WHISPER
# ==========================================================

print("\nLoading Whisper...\n")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="float32"
)
print("Whisper Loaded.\n")

# ==========================================================
# LOAD SILERO VAD
# ==========================================================

print("Loading Silero VAD...")

vad_model = load_silero_vad()

vad_iterator = VADIterator(
    vad_model,
    threshold=VAD_THRESHOLD,
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
    speech_pad_ms=300
)

print("Silero VAD Loaded.\n")
# ==========================================================
# AUDIO QUEUE
# ==========================================================

audio_queue = queue.Queue(maxsize=200)


def audio_callback(indata, frames, time_info, status):

    if status:
        print("Audio:", status)

    try:
        audio_queue.put_nowait(indata.copy())

    except queue.Full:
        # Drop the oldest block instead of allowing
        # the audio system to become overloaded.
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

# ==========================================================
# RECORD ONE UTTERANCE - STREAMING SILERO VAD
# ==========================================================

def record_until_silence():

    print("\nListening...\n")

    pre_buffer = []
    recording = []

    speech_started = False
    recording_start = None

    # Time when Silero first tells us speech has ended
    silence_start = None

    pre_buffer_blocks = max(
        1,
        int(PRE_BUFFER_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
    )

    # Reset Silero streaming state
    vad_iterator.reset_states()

    while True:

        try:
            data = audio_queue.get(timeout=1)

        except queue.Empty:
            continue

        # --------------------------------------------------
        # Convert microphone data to mono float32
        # --------------------------------------------------

        audio_block = data[:, 0].astype(np.float32)

        # --------------------------------------------------
        # Maintain pre-buffer
        # --------------------------------------------------

        if not speech_started:

            pre_buffer.append(audio_block.copy())

            if len(pre_buffer) > pre_buffer_blocks:
                pre_buffer.pop(0)

        # --------------------------------------------------
        # Run Silero VAD
        # --------------------------------------------------

        audio_tensor = torch.from_numpy(audio_block)

        speech_event = vad_iterator(audio_tensor)

        # ==================================================
        # WAITING FOR SPEECH
        # ==================================================

        if not speech_started:

            if speech_event is not None:

                if "start" in speech_event:

                    print("Speech detected.")

                    speech_started = True

                    recording_start = time.time()

                    silence_start = None

                    # Include audio immediately before speech
                    recording.extend(pre_buffer)

                    pre_buffer.clear()

            continue

        # ==================================================
        # SPEECH HAS STARTED
        # ==================================================

        recording.append(audio_block.copy())

        # ==================================================
        # SPEECH CONTINUES
        # ==================================================

        if speech_event is not None:

            if "start" in speech_event:

                # User started speaking again
                silence_start = None

                continue

            # ==================================================
            # SILERO DETECTED END OF SPEECH
            # ==================================================

            if "end" in speech_event:

                # IMPORTANT:
                # Do NOT immediately stop recording.
                #
                # Start a silence timer instead.
                if silence_start is None:

                    silence_start = time.time()

                    print("Possible speech end - waiting...")

        # ==================================================
        # POST-SPEECH SILENCE CHECK
        # ==================================================

        if silence_start is not None:

            silence_duration = time.time() - silence_start

            if silence_duration >= POST_BUFFER_SECONDS:

                print("Post-speech buffer complete.")

                break

        # ==================================================
        # SAFETY TIMEOUT
        # ==================================================

        if recording_start is not None:

            elapsed = time.time() - recording_start

            if elapsed >= MAX_RECORDING_SECONDS:

                print("Maximum recording time reached.")

                break

    # ======================================================
    # SAFETY CHECK
    # ======================================================

    if not recording:

        return None

    # ======================================================
    # COMBINE AUDIO
    # ======================================================

    audio = np.concatenate(
        recording
    ).astype(np.float32)

    # ======================================================
    # REMOVE DC OFFSET
    # ======================================================

    audio = audio - np.mean(audio)

    # ======================================================
    # NORMALIZE
    # ======================================================

    peak = np.max(np.abs(audio))

    if peak > 0.001:

        audio = audio / peak

        audio = audio * 0.95

    # ======================================================
    # SAVE WAV
    # ======================================================

    filename = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ).name

    sf.write(
        filename,
        audio,
        SAMPLE_RATE
    )

    print("Recording complete.")

    return filename

# ==========================================================
# SPEECH TO TEXT
# ==========================================================

def speech_to_text(filename):

    try:

        segments, info = model.transcribe(

            filename,

            language="en",

            beam_size=5,

            temperature=0.0,

            vad_filter=True,

            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=300
            ),

            condition_on_previous_text=False,

            # Improve recognition consistency
            compression_ratio_threshold=2.4,

            log_prob_threshold=-1.0,

            no_speech_threshold=0.6
        )

        text = ""

        for segment in segments:

            text += segment.text

        return text.strip()

    finally:

        if os.path.exists(filename):

            os.remove(filename)

# ==========================================================
# WAKE WORD
# ==========================================================

def detect_wake_word(text):

    text_lower = text.lower().strip()

    # ------------------------------------------------------
    # Normalize punctuation
    # ------------------------------------------------------

    normalized = (
        text_lower
        .replace(",", " ")
        .replace(".", " ")
        .replace("?", " ")
        .replace("!", " ")
        .replace(";", " ")
        .replace(":", " ")
    )

    words = normalized.split()

    # ------------------------------------------------------
    # EXACT "SAM" WORD MATCH
    # ------------------------------------------------------

    if "sam" in words:

        print("Wake word exact match : sam")

        return True

    # ------------------------------------------------------
    # Check longer known variants
    # ------------------------------------------------------

    for wake_word in WAKE_VARIANTS:

        if wake_word == "sam":
            continue

        if wake_word in normalized:

            print("Wake word variant match :", wake_word)

            return True

    # ------------------------------------------------------
    # Fuzzy matching ONLY against individual words
    #
    # This prevents the short word "sam" from matching
    # arbitrary parts of the whole sentence.
    # ------------------------------------------------------

    best_word = None
    best_score = 0

    for word in words:

        # Ignore very short words
        if len(word) < 3:
            continue

        score = fuzz.ratio("sam", word)

        if score > best_score:

            best_score = score
            best_word = word

    print(
        "Best wake-word candidate :",
        best_word,
        best_score
    )

    # ------------------------------------------------------
    # Conservative fuzzy match
    # ------------------------------------------------------

    if best_score >= 80:

        print("Fuzzy wake word detected.")

        return True

    return False

# ==========================================================
# REMOVE WAKE WORD
# ==========================================================

def remove_wake(text):

    original_text = text.strip()

    # ------------------------------------------------------
    # Normalize punctuation for word detection
    # ------------------------------------------------------

    words = original_text.split()

    # ------------------------------------------------------
    # Remove exact wake word wherever it appears
    # ------------------------------------------------------

    for i, word in enumerate(words):

        clean_word = word.strip(
            " ,.?;:!"
        ).lower()

        if clean_word == "sam":

            print("Wake word removed : sam")

            words.pop(i)

            command = " ".join(words)

            return command.strip(
                " ,.?;:!"
            )

    # ------------------------------------------------------
    # Remove longer known variants
    # ------------------------------------------------------

    lower_text = original_text.lower()

    for wake_word in WAKE_VARIANTS:

        if wake_word == "sam":
            continue

        position = lower_text.find(wake_word)

        if position != -1:

            print(
                "Wake word variant removed :",
                wake_word
            )

            before = original_text[:position]

            after = original_text[
                position + len(wake_word):
            ]

            command = (
                before + " " + after
            ).strip()

            command = command.strip(
                " ,.?;:!"
            )

            command = " ".join(
                command.split()
            )

            return command

    return original_text

# ==========================================================
# PLACEHOLDER
# ==========================================================

def process_command(command):

    print("\nUser:", command)

    current_time = datetime.now()

    today = current_time.strftime("%A, %d %B %Y")

    clock = current_time.strftime("%I:%M %p")

    system_prompt = """
You are Samwise.

You are an intelligent English voice assistant for a smart home automation system.

Rules:

1. Keep replies short.

2. Be friendly.

3. Never mention OpenAI, Groq, Gemini or AI models.

4. Answer naturally.

5. If the command is related to controlling lights, fans, doors,
AC, curtains or other smart devices,
respond with

[HOME]
followed by the command.

Otherwise answer normally.
"""

    
    try:

        conversation.append(
            {
                "role": "user",
                "content": command
            }
        )

        completion = groq_client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=conversation,

            temperature=0.3,

            max_tokens=300

        )

        answer = completion.choices[0].message.content

        conversation.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        print("\nSam:\n")
        print(answer)

        if len(conversation) > 20:
            conversation[:] = [conversation[0]] + conversation[-19:]

        return

    except Exception as e:

        print("\nGroq failed.")
        print(e)
            
    try:

        history = ""

        for msg in conversation:
            history += f"{msg['role']}: {msg['content']}\n"

        response = gemini_client.models.generate_content(

            model="gemini-2.5-flash",

            contents=history

        )

        answer = response.text

        conversation.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        print("\nGemini Fallback:\n")
        print(answer)

        if len(conversation) > 20:
            conversation[:] = [conversation[0]] + conversation[-19:]

    except Exception as e:

        print("\nGemini also failed.")
        print(e)

# ==========================================================
# START AUDIO STREAM
# ==========================================================

stream = sd.InputStream(

    samplerate=SAMPLE_RATE,

    channels=CHANNELS,

    blocksize=BLOCK_SIZE,

    callback=audio_callback,

    device=MIC_DEVICE

)

stream.start()

print("========================================")
print("      SAM READY")
print("========================================")

# ==========================================================
# MAIN LOOP
# ==========================================================

while True:

    try:

        # --------------------------------------------------
        # Record one complete utterance
        # --------------------------------------------------

        wav = record_until_silence()

        if wav is None:
            continue

        # --------------------------------------------------
        # Speech to text
        # --------------------------------------------------

        text = speech_to_text(wav)

        if len(text) == 0:
            continue

        print("\nRecognized :")
        print(text)
        print()

        # ==================================================
        # CHECK CONVERSATION TIMEOUT
        # ==================================================

        current_time = time.time()

        if conversation_active:

            if current_time - last_command_time > WAKE_TIMEOUT:

                conversation_active = False

                print("Conversation mode ended.")
                print("Wake word required again.\n")

        # ==================================================
        # ACTIVE CONVERSATION
        # ==================================================

        if conversation_active:

            print("Conversation Mode Active\n")

            # ----------------------------------------------
            # During active conversation, EVERYTHING
            # recognized is treated as a command.
            # ----------------------------------------------

            command = text.strip()

            if command != "":

                process_command(command)

                # ------------------------------------------
                # Restart the 10-second timer
                # ------------------------------------------

                last_command_time = time.time()

            continue

        # ==================================================
        # CONVERSATION IS NOT ACTIVE
        # ==================================================
        #
        # Only now do we check for the wake word.
        #
        # ==================================================

        wake_detected = detect_wake_word(text)

        if wake_detected:

            print("Wake Word Detected\n")

            # ----------------------------------------------
            # Start conversation mode
            # ----------------------------------------------

            conversation_active = True

            # ----------------------------------------------
            # Remove wake word from anywhere in sentence
            # ----------------------------------------------

            command = remove_wake(text)

            # ----------------------------------------------
            # Remove common filler words
            # ----------------------------------------------

            for word in [
                "is",
                "please",
                "can you",
                "could you"
            ]:

                if command.lower().startswith(word):

                    command = command[len(word):].strip()

            # ----------------------------------------------
            # Wake word only
            # ----------------------------------------------

            if command == "":

                print("Yes?\n")

                last_command_time = time.time()

                continue

            # ----------------------------------------------
            # Wake word + command
            # ----------------------------------------------

            process_command(command)

            # ----------------------------------------------
            # Restart 10-second timer
            # ----------------------------------------------

            last_command_time = time.time()

        else:

            # ------------------------------------------------
            # No wake word and no active conversation
            # ------------------------------------------------

            print("Ignored.\n")


    except KeyboardInterrupt:

        print("\n\nSam stopped.")

        break


    except Exception as e:

        print("\nRuntime error:")
        print(e)

        print("\nContinuing...\n")

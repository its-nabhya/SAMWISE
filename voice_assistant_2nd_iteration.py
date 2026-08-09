
import queue
import tempfile
import os
import time
import asyncio
import io

import numpy as np
import sounddevice as sd
import soundfile as sf

from faster_whisper import WhisperModel
from rapidfuzz import fuzz

from groq import Groq
from google import genai

import torch
from silero_vad import load_silero_vad, VADIterator

import edge_tts



# ==========================================================
# API KEYS
# ==========================================================

GROQ_API_KEY = "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"

GEMINI_API_KEY = "AQ.Ab8RN6LfH2VURtpZiPbSUCxSoJg5DcF9hzLCm5tZ-Se6Iwimig"


# ==========================================================
# SETTINGS
# ==========================================================

MIC_DEVICE = 1

# Bluetooth speaker:
#
# None = use Windows default output device.
#
# If your Bluetooth speaker is the Windows default speaker,
# leave this as None.
#
# If you know its SoundDevice index, put that number here.
#
# Example:
# SPEAKER_DEVICE = 5
#
SPEAKER_DEVICE = None


SAMPLE_RATE = 16000
CHANNELS = 1

# 512 = 32 ms at 16 kHz
BLOCK_SIZE = 512


# ==========================================================
# RECORDING SETTINGS
# ==========================================================

PRE_BUFFER_SECONDS = 0.35

# CHANGED:
# Reduced from 1.3 seconds.
# Sam now stops recording sooner after you finish speaking.
POST_BUFFER_SECONDS = 0.80

# Maximum utterance length
MAX_RECORDING_SECONDS = 12


# ==========================================================
# NOISE / VAD SETTINGS
# ==========================================================

# CHANGED:
# Higher threshold means less sensitivity to background noise.
VAD_THRESHOLD = 0.65

# CHANGED:
# Silero needs less continuous silence before reporting end.
MIN_SILENCE_DURATION_MS = 500

# CHANGED:
# Simple RMS gate.
#
# If microphone signal is extremely quiet, we don't allow
# Silero VAD to start recording.
#
# You may adjust this:
#
# 0.003 = very sensitive
# 0.005 = sensitive
# 0.008 = normal
# 0.012 = less sensitive
#
MIN_RMS = 0.008


# ==========================================================
# CONVERSATION SETTINGS
# ==========================================================

WAKE_TIMEOUT = 15.0

# CHANGED:
# Ignore microphone for a short time after Sam finishes speaking.
# This prevents Bluetooth/room echo from immediately triggering
# another recording.
POST_TTS_COOLDOWN = 0.45


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
# TEXT TO SPEECH
# ==========================================================

TTS_VOICE = "en-US-GuyNeural"

TTS_RATE = "+5%"
TTS_VOLUME = "+0%"


# ==========================================================
# AUDIO QUEUE
# ==========================================================

audio_queue = queue.Queue(maxsize=200)


def clear_audio_queue():

    while True:

        try:
            audio_queue.get_nowait()

        except queue.Empty:
            break


# ==========================================================
# AUDIO CALLBACK
# ==========================================================

def audio_callback(indata, frames, time_info, status):

    if status:
        print("Audio:", status)

    try:

        audio_queue.put_nowait(
            indata.copy()
        )

    except queue.Full:

        try:
            audio_queue.get_nowait()

        except queue.Empty:
            pass

        try:
            audio_queue.put_nowait(
                indata.copy()
            )

        except queue.Full:
            pass


# ==========================================================
# MICROPHONE STREAM
# ==========================================================

stream = None


# ==========================================================
# TEXT TO SPEECH
# ==========================================================

def speak(text):

    global stream

    if not text:
        return

    # ------------------------------------------------------
    # Remove HOME protocol markers from spoken response
    # ------------------------------------------------------

    spoken_text = text

    if "[HOME]" in spoken_text:

        lines = spoken_text.splitlines()

        spoken_lines = [
            line
            for line in lines
            if line.strip() != "[HOME]"
        ]

        spoken_text = " ".join(
            spoken_lines
        )

    spoken_text = spoken_text.strip()

    if not spoken_text:
        return

    # ======================================================
    # CHANGED:
    # STOP MICROPHONE BEFORE GENERATING TTS
    #
    # This is very important.
    #
    # Previously the microphone could remain active while
    # Edge-TTS was generating audio.
    #
    # Now the complete response cycle is:
    #
    # STOP MIC
    #     ↓
    # GENERATE TTS
    #     ↓
    # PLAY TTS
    #     ↓
    # CLEAR QUEUE
    #     ↓
    # START MIC
    #
    # Sam therefore cannot hear his own response.
    # ======================================================

    try:

        if stream is not None:

            try:
                stream.stop()
            except Exception:
                pass

        # Remove anything captured before stopping.
        clear_audio_queue()

        print("\nSpeaking...")

        # --------------------------------------------------
        # Generate TTS
        # --------------------------------------------------

        async def generate():

            communicate = edge_tts.Communicate(
                spoken_text,
                TTS_VOICE,
                rate=TTS_RATE,
                volume=TTS_VOLUME
            )

            audio_data = bytearray()

            async for chunk in communicate.stream():

                if chunk["type"] == "audio":

                    audio_data.extend(
                        chunk["data"]
                    )

            return bytes(audio_data)

        mp3_data = asyncio.run(
            generate()
        )

        if not mp3_data:

            print("TTS produced no audio.")

            return

        # --------------------------------------------------
        # Decode MP3 directly in memory
        # --------------------------------------------------

        import av

        container = av.open(
            io.BytesIO(mp3_data),
            format="mp3"
        )

        audio_frames = []
        sample_rate = None

        for frame in container.decode(audio=0):

            array = frame.to_ndarray()

            audio_frames.append(array)

            if sample_rate is None:

                sample_rate = frame.sample_rate

        container.close()

        if not audio_frames:

            print("Could not decode TTS audio.")

            return

        # --------------------------------------------------
        # Combine decoded audio
        # --------------------------------------------------

        audio = np.concatenate(
            audio_frames,
            axis=1
        )

        # --------------------------------------------------
        # Convert to mono
        # --------------------------------------------------

        if audio.ndim > 1:

            audio = np.mean(
                audio,
                axis=0
            )

        # --------------------------------------------------
        # Float32
        # --------------------------------------------------

        audio = audio.astype(
            np.float32
        )

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        peak = np.max(
            np.abs(audio)
        )

        if peak > 0:

            audio = (
                audio / peak
            ) * 0.90

        # ==================================================
        # DIRECT AUDIO PLAYBACK
        # ==================================================
        #
        # This DOES NOT open Windows Media Player.
        #
        # sounddevice sends the samples directly to the
        # selected Windows audio device.
        #
        # If SPEAKER_DEVICE = None:
        #     Windows default output is used.
        #
        # Therefore set your Bluetooth speaker as the
        # Windows default output device.
        #
        # ==================================================

        sd.play(
            audio,
            samplerate=sample_rate,
            device=SPEAKER_DEVICE,
            blocking=True
        )

        sd.stop()

        print("Finished speaking.\n")

    except Exception as e:

        print("\nTTS Error:")
        print(type(e).__name__)
        print(e)

    finally:

        # ==================================================
        # CHANGED:
        # ALWAYS CLEAR OLD MIC AUDIO
        # ==================================================

        clear_audio_queue()

        # ==================================================
        # CHANGED:
        # SHORT COOLDOWN AFTER SPEAKING
        # ==================================================

        time.sleep(
            POST_TTS_COOLDOWN
        )

        # ==================================================
        # RESTART MICROPHONE
        # ==================================================

        try:

            if stream is not None:

                stream.start()

        except Exception as e:

            print(
                "Could not restart microphone:"
            )

            print(e)

        clear_audio_queue()


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
- Give one sentence answers ONLY.
- Do not repeat the user's question.
- Do not continue speaking after answering.
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
    "base.en",
    device="cpu",
    compute_type="int8"
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

    min_silence_duration_ms=
        MIN_SILENCE_DURATION_MS,

    speech_pad_ms=250
)


print("Silero VAD Loaded.\n")


# ==========================================================
# RECORD ONE UTTERANCE
# ==========================================================

def record_until_silence():

    print("\nListening...\n")

    pre_buffer = []

    recording = []

    speech_started = False

    recording_start = None

    silence_start = None

    pre_buffer_blocks = max(

        1,

        int(
            PRE_BUFFER_SECONDS
            * SAMPLE_RATE
            / BLOCK_SIZE
        )

    )


    # ------------------------------------------------------
    # Reset VAD
    # ------------------------------------------------------

    vad_iterator.reset_states()


    while True:

        try:

            data = audio_queue.get(
                timeout=1
            )

        except queue.Empty:

            continue


        # --------------------------------------------------
        # Convert microphone data to mono
        # --------------------------------------------------

        audio_block = (
            data[:, 0]
            .astype(np.float32)
        )


        # ==================================================
        # CHANGED:
        # RMS NOISE GATE
        # ==================================================
        #
        # Prevent very quiet background noise from being
        # treated as speech.
        #
        # ==================================================

        rms = np.sqrt(
            np.mean(
                audio_block ** 2
            )
        )


        # --------------------------------------------------
        # Maintain pre-buffer
        # --------------------------------------------------

        if not speech_started:

            pre_buffer.append(
                audio_block.copy()
            )

            if (
                len(pre_buffer)
                > pre_buffer_blocks
            ):

                pre_buffer.pop(0)


        # ==================================================
        # NOISE GATE
        # ==================================================

        if (
            not speech_started
            and rms < MIN_RMS
        ):

            # Do not send extremely quiet audio to VAD.
            continue


        # --------------------------------------------------
        # Run Silero VAD
        # --------------------------------------------------

        audio_tensor = torch.from_numpy(
            audio_block
        )


        speech_event = vad_iterator(
            audio_tensor
        )


        # ==================================================
        # WAITING FOR SPEECH
        # ==================================================

        if not speech_started:

            if speech_event is not None:

                if "start" in speech_event:

                    print(
                        "Speech detected."
                    )

                    speech_started = True

                    recording_start = (
                        time.time()
                    )

                    silence_start = None

                    recording.extend(
                        pre_buffer
                    )

                    pre_buffer.clear()

            continue


        # ==================================================
        # SPEECH HAS STARTED
        # ==================================================

        recording.append(
            audio_block.copy()
        )


        # ==================================================
        # SPEECH CONTINUES
        # ==================================================

        if speech_event is not None:

            if "start" in speech_event:

                silence_start = None

                continue


            # ==================================================
            # SILERO DETECTED END
            # ==================================================

            if "end" in speech_event:

                if silence_start is None:

                    silence_start = (
                        time.time()
                    )

                    print(
                        "Possible speech end - waiting..."
                    )


        # ==================================================
        # POST SPEECH SILENCE
        # ==================================================

        if silence_start is not None:

            silence_duration = (
                time.time()
                - silence_start
            )


            if (
                silence_duration
                >= POST_BUFFER_SECONDS
            ):

                print(
                    "Post-speech buffer complete."
                )

                break


        # ==================================================
        # MAXIMUM RECORDING TIME
        # ==================================================

        if recording_start is not None:

            elapsed = (
                time.time()
                - recording_start
            )


            if (
                elapsed
                >= MAX_RECORDING_SECONDS
            ):

                print(
                    "Maximum recording time reached."
                )

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

    audio = (
        audio
        - np.mean(audio)
    )


    # ======================================================
    # CHECK AUDIO LEVEL
    # ======================================================

    rms = np.sqrt(
        np.mean(
            audio ** 2
        )
    )


    # CHANGED:
    # Reject recordings that are basically noise/silence.

    if rms < MIN_RMS:

        print(
            "Audio too quiet - ignored."
        )

        return None


    # ======================================================
    # NORMALIZE
    # ======================================================

    peak = np.max(
        np.abs(audio)
    )


    if peak > 0.001:

        audio = (
            audio / peak
        ) * 0.90


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


    print(
        "Recording complete."
    )


    return filename


# ==========================================================
# SPEECH TO TEXT
# ==========================================================

def speech_to_text(filename):

    try:

        segments, info = model.transcribe(

            filename,

            language="en",

            # CHANGED:
            # Faster than beam_size=5.
            beam_size=1,

            temperature=0.0,

            vad_filter=True,

            vad_parameters=dict(

                min_silence_duration_ms=400,

                speech_pad_ms=200

            ),

            condition_on_previous_text=False,

            compression_ratio_threshold=2.4,

            log_prob_threshold=-1.0,

            # CHANGED:
            # Higher value helps reject silence/noise.
            no_speech_threshold=0.70
        )


        text_parts = []


        for segment in segments:

            # --------------------------------------------------
            # CHANGED:
            # Ignore extremely low-confidence segments.
            # --------------------------------------------------

            if (
                hasattr(segment, "no_speech_prob")
                and segment.no_speech_prob > 0.75
            ):

                continue


            text_parts.append(
                segment.text
            )


        text = " ".join(
            text_parts
        ).strip()


        # ======================================================
        # CHANGED:
        # Ignore extremely short/noisy transcription
        # ======================================================

        if len(text.strip()) < 2:

            return ""


        return text


    finally:

        if os.path.exists(filename):

            try:
                os.remove(filename)
            except Exception:
                pass


# ==========================================================
# WAKE WORD
# ==========================================================

def detect_wake_word(text):

    text_lower = (
        text.lower()
        .strip()
    )


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


    # ======================================================
    # EXACT SAM
    # ======================================================

    if "sam" in words:

        print(
            "Wake word exact match : sam"
        )

        return True


    # ======================================================
    # KNOWN VARIANTS
    # ======================================================

    for wake_word in WAKE_VARIANTS:

        if wake_word == "sam":

            continue


        if wake_word in words:

            print(
                "Wake word variant match :",
                wake_word
            )

            return True


    # ======================================================
    # FUZZY MATCH
    # ======================================================

    best_word = None

    best_score = 0


    for word in words:

        if len(word) < 3:

            continue


        score = fuzz.ratio(
            "sam",
            word
        )


        if score > best_score:

            best_score = score

            best_word = word


    print(
        "Best wake-word candidate :",
        best_word,
        best_score
    )


    # ======================================================
    # CHANGED:
    # More conservative fuzzy threshold
    # ======================================================

    if best_score >= 85:

        print(
            "Fuzzy wake word detected."
        )

        return True


    return False


# ==========================================================
# REMOVE WAKE WORD
# ==========================================================

def remove_wake(text):

    original_text = (
        text.strip()
    )


    words = original_text.split()


    # ======================================================
    # EXACT SAM
    # ======================================================

    for i, word in enumerate(words):

        clean_word = word.strip(
            " ,.?;:!"
        ).lower()


        if clean_word == "sam":

            print(
                "Wake word removed : sam"
            )


            words.pop(i)


            command = " ".join(
                words
            )


            return command.strip(
                " ,.?;:!"
            )


    # ======================================================
    # VARIANTS
    # ======================================================

    lower_text = (
        original_text.lower()
    )


    for wake_word in WAKE_VARIANTS:

        if wake_word == "sam":

            continue


        # Only match complete words.
        normalized_words = (
            lower_text.split()
        )


        if wake_word in normalized_words:

            position = lower_text.find(
                wake_word
            )


            print(
                "Wake word variant removed :",
                wake_word
            )


            before = (
                original_text[:position]
            )


            after = original_text[
                position
                + len(wake_word):
            ]


            command = (
                before
                + " "
                + after
            ).strip()


            command = command.strip(
                " ,.?;:!"
            )


            return " ".join(
                command.split()
            )


    return original_text


# ==========================================================
# PROCESS COMMAND
# ==========================================================

def process_command(command):

    print(
        "\nUser:",
        command
    )


    try:

        conversation.append(

            {
                "role": "user",
                "content": command
            }

        )


        completion = (
            groq_client
            .chat
            .completions
            .create(

                model=
                    "llama-3.3-70b-versatile",

                messages=conversation,

                temperature=0.3,

                max_tokens=100
            )
        )


        answer = (
            completion
            .choices[0]
            .message
            .content
        )


        conversation.append(

            {
                "role": "assistant",
                "content": answer
            }

        )


        print(
            "\nSam:\n"
        )

        print(answer)


        # ==================================================
        # IMPORTANT:
        # speak() stops the microphone BEFORE TTS.
        # ==================================================

        speak(answer)


        if len(conversation) > 20:

            conversation[:] = (
                [conversation[0]]
                + conversation[-19:]
            )


        return


    except Exception as e:

        print(
            "\nGroq failed."
        )

        print(e)


    # ======================================================
    # GEMINI FALLBACK
    # ======================================================

    try:

        history = ""


        for msg in conversation:

            history += (
                f"{msg['role']}: "
                f"{msg['content']}\n"
            )


        response = (
            gemini_client
            .models
            .generate_content(

                model="gemini-2.5-flash",

                contents=history
            )
        )


        answer = response.text


        conversation.append(

            {
                "role": "assistant",
                "content": answer
            }

        )


        print(
            "\nGemini Fallback:\n"
        )

        print(answer)


        speak(answer)


        if len(conversation) > 20:

            conversation[:] = (
                [conversation[0]]
                + conversation[-19:]
            )


    except Exception as e:

        print(
            "\nGemini also failed."
        )

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


print(
    "========================================"
)

print(
    "      SAM READY"
)

print(
    "========================================"
)


# ==========================================================
# MAIN LOOP
# ==========================================================

while True:

    try:

        # ==================================================
        # RECORD
        # ==================================================

        wav = (
            record_until_silence()
        )


        if wav is None:

            continue


        # ==================================================
        # TRANSCRIBE
        # ==================================================

        text = speech_to_text(
            wav
        )


        if len(text) == 0:

            continue


        print(
            "\nRecognized :"
        )

        print(text)

        print()


        # ==================================================
        # CONVERSATION TIMEOUT
        # ==================================================

        current_time = (
            time.time()
        )


        if conversation_active:

            if (
                current_time
                - last_command_time
                > WAKE_TIMEOUT
            ):

                conversation_active = False


                print(
                    "Conversation mode ended."
                )

                print(
                    "Wake word required again.\n"
                )


        # ==================================================
        # ACTIVE CONVERSATION
        # ==================================================

        if conversation_active:

            print(
                "Conversation Mode Active\n"
            )


            command = (
                text.strip()
            )


            if command != "":

                process_command(
                    command
                )


                last_command_time = (
                    time.time()
                )


            continue


        # ==================================================
        # NO ACTIVE CONVERSATION
        # ==================================================

        wake_detected = (
            detect_wake_word(text)
        )


        if wake_detected:

            print(
                "Wake Word Detected\n"
            )


            conversation_active = True


            command = remove_wake(
                text
            )


            # ==================================================
            # Remove common filler words
            # ==================================================

            fillers = [

                "is",
                "please",
                "can you",
                "could you"

            ]


            for word in fillers:

                if (
                    command
                    .lower()
                    .startswith(word)
                ):

                    command = (
                        command[
                            len(word):
                        ]
                        .strip()
                    )


            # ==================================================
            # Wake word only
            # ==================================================

            if command == "":

                print(
                    "Yes?\n"
                )


                last_command_time = (
                    time.time()
                )


                continue


            # ==================================================
            # WAKE WORD + COMMAND
            # ==================================================

            process_command(
                command
            )


            last_command_time = (
                time.time()
            )


        else:

            print(
                "Ignored.\n"
            )


    # ======================================================
    # CTRL+C
    # ======================================================

    except KeyboardInterrupt:

        print(
            "\n\nSam stopped."
        )

        break


    # ======================================================
    # RUNTIME ERROR
    # ======================================================

    except Exception as e:

        print(
            "\nRuntime error:"
        )

        print(e)

        print(
            "\nContinuing...\n"
        )


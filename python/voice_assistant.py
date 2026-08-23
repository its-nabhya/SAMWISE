"""
Basic SAM Voice Assistant for Arduino UNO Q

Pipeline:

USB Microphone
      ↓
RMS Voice Detection
      ↓
Native microphone sampling rate
      ↓
Resample to 16 kHz
      ↓
Faster-Whisper
      ↓
SAM wake word
      ↓
Gemini
      ↓
Groq fallback
      ↓
Edge TTS

Required libraries:

    sounddevice
    soundfile
    numpy
    faster-whisper
    google-genai
    groq
    edge-tts

Environment variables:

    GEMINI_API_KEY
    GROQ_API_KEY
"""

import asyncio
import difflib
import io
import os
import queue
import time

from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf

from faster_whisper import WhisperModel

from google import genai
from groq import Groq

import edge_tts


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# API KEYS
# ------------------------------------------------------------

GROQ_API_KEY = "ENTER YOUR GROQ API KEY HERE"  # Replace with your GROQ API key
GEMINI_API_KEY = "ENTER YOUR GEMINI API KEY HERE"  # Replace with your GEMINI API key


# ============================================================
# MICROPHONE
# ============================================================

# Your UNO Q shows:
#
# 0 : USB PnP Sound Device: Audio (hw:0,0)
#
# Therefore use device index 0.

MIC_DEVICE = 0


# ============================================================
# SPEAKER
# ============================================================

# None = default output device

SPEAKER_DEVICE = None


# ============================================================
# WHISPER AUDIO FORMAT
# ============================================================

# Whisper works best with 16 kHz mono.

WHISPER_SAMPLE_RATE = 16000

CHANNELS = 1


# ============================================================
# MICROPHONE SETTINGS
# ============================================================

# This is determined automatically from the USB microphone.

MIC_SAMPLE_RATE = None


# ============================================================
# AUDIO BLOCK
# ============================================================

# This is deliberately moderate.

BLOCK_SIZE = 1024


# ============================================================
# VOICE ACTIVITY DETECTION
# ============================================================

# Adjusted slightly lower because USB microphones can have
# relatively low RMS levels.

MIN_RMS = 0.008

SILENCE_RMS = 0.006


# ============================================================
# RECORDING
# ============================================================

PRE_BUFFER_SECONDS = 0.30

SILENCE_DURATION = 0.90

MAX_RECORDING_SECONDS = 8.0

MIN_RECORDING_SECONDS = 0.30


# ============================================================
# WHISPER
# ============================================================

WHISPER_MODEL = "base.en"

WHISPER_DEVICE = "cpu"

WHISPER_COMPUTE_TYPE = "int8"

WHISPER_CPU_THREADS = 2

WHISPER_NUM_WORKERS = 1


# ============================================================
# WAKE WORD
# ============================================================

WAKE_WORD = "sam"


WAKE_VARIANTS = {
    "sam",
    "samm",
    "saam",
    "samh",
    "ssam",
    "som",
    "sang",
    "sam.",
    "sam,"
}


WAKE_SIMILARITY = 0.72


# ============================================================
# AI MODELS
# ============================================================

GEMINI_MODEL = "gemini-3.6-flash"

GROQ_MODEL = "qwen/qwen3.6-27b"


# ============================================================
# TTS
# ============================================================

TTS_VOICE = "en-US-GuyNeural"

TTS_RATE = "+5%"

TTS_VOLUME = "+0%"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Sam, a simple voice assistant.

Answer the user's question directly and naturally.

Keep responses short and conversational.

Do not repeat the user's question.

Do not give long explanations unless the user specifically asks.

You are running on an Arduino UNO Q based smart system.
""".strip()


# ============================================================
# AUDIO MANAGER
# ============================================================

class AudioManager:

    def __init__(self):

        self.stream = None

        self.audio_queue = queue.Queue(maxsize=100)

        self.sample_rate = None

        self.pre_buffer_blocks = 1


    # ========================================================
    # CLEAR QUEUE
    # ========================================================

    def clear_queue(self):

        while True:

            try:

                self.audio_queue.get_nowait()

            except queue.Empty:

                break


    # ========================================================
    # AUDIO CALLBACK
    # ========================================================

    def callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):

        if status:

            print(
                "Audio status:",
                status
            )

        try:

            self.audio_queue.put_nowait(
                indata.copy()
            )

        except queue.Full:

            try:

                self.audio_queue.get_nowait()

            except queue.Empty:

                pass

            try:

                self.audio_queue.put_nowait(
                    indata.copy()
                )

            except queue.Full:

                pass


    # ========================================================
    # LIST AUDIO DEVICES
    # ========================================================

    def print_devices(self):

        print("--------------------------------")
        print("AVAILABLE AUDIO DEVICES")
        print("--------------------------------")

        try:

            devices = sd.query_devices()

            default_input = sd.default.device[0]

            default_output = sd.default.device[1]

            print(
                "Default input :",
                default_input
            )

            print(
                "Default output:",
                default_output
            )

            print("--------------------------------")

            for index, device in enumerate(devices):

                print(
                    index,
                    ":",
                    device["name"],
                    "| IN:",
                    device["max_input_channels"],
                    "| OUT:",
                    device["max_output_channels"],
                    "| RATE:",
                    device["default_samplerate"]
                )

            print("--------------------------------")

        except Exception as e:

            print(
                "Could not list audio devices:",
                type(e).__name__,
                e
            )


    # ========================================================
    # DETERMINE MICROPHONE SAMPLE RATE
    # ========================================================

    def determine_sample_rate(self):

        print("--------------------------------")
        print("DETECTING USB MICROPHONE")
        print("--------------------------------")

        try:

            device = sd.query_devices(
                MIC_DEVICE,
                "input"
            )

            self.sample_rate = int(
                device["default_samplerate"]
            )

            print(
                "Device:",
                device["name"]
            )

            print(
                "Device index:",
                MIC_DEVICE
            )

            print(
                "Native sample rate:",
                self.sample_rate
            )

            print(
                "Whisper sample rate:",
                WHISPER_SAMPLE_RATE
            )

            print(
                "Channels:",
                CHANNELS
            )

            print("--------------------------------")

            return True

        except Exception as e:

            print("--------------------------------")
            print("MICROPHONE ERROR")
            print("--------------------------------")

            print(
                type(e).__name__,
                ":",
                e
            )

            print("--------------------------------")

            return False


    # ========================================================
    # CHECK MICROPHONE
    # ========================================================

    def check_microphone(self):

        if self.sample_rate is None:

            if not self.determine_sample_rate():

                return False


        print("--------------------------------")
        print("CHECKING USB MICROPHONE")
        print("--------------------------------")

        try:

            sd.check_input_settings(

                device=MIC_DEVICE,

                channels=CHANNELS,

                samplerate=self.sample_rate,

                dtype="float32"
            )

            print(
                "MICROPHONE READY"
            )

            print("--------------------------------")

            return True

        except Exception as e:

            print("--------------------------------")
            print("MICROPHONE ERROR")
            print("--------------------------------")

            print(
                type(e).__name__,
                ":",
                e
            )

            print("--------------------------------")

            return False


    # ========================================================
    # START MICROPHONE
    # ========================================================

    def start(self):

        if self.stream is not None:

            return


        if self.sample_rate is None:

            if not self.determine_sample_rate():

                raise RuntimeError(
                    "Could not determine microphone sample rate."
                )


        self.clear_queue()


        self.pre_buffer_blocks = max(
            1,
            int(
                PRE_BUFFER_SECONDS
                * self.sample_rate
                / BLOCK_SIZE
            )
        )


        print("--------------------------------")
        print(
            "Starting microphone at",
            self.sample_rate,
            "Hz"
        )
        print("--------------------------------")


        self.stream = sd.InputStream(

            device=MIC_DEVICE,

            samplerate=self.sample_rate,

            channels=CHANNELS,

            blocksize=BLOCK_SIZE,

            dtype="float32",

            callback=self.callback
        )


        self.stream.start()


        print(
            "Microphone started."
        )


    # ========================================================
    # STOP MICROPHONE
    # ========================================================

    def stop(self):

        if self.stream is None:

            return

        try:

            self.stream.stop()

        except Exception:

            pass


    # ========================================================
    # CLOSE MICROPHONE
    # ========================================================

    def close(self):

        if self.stream is not None:

            try:

                self.stream.stop()

            except Exception:

                pass

            try:

                self.stream.close()

            except Exception:

                pass

        self.stream = None

        self.clear_queue()


    # ========================================================
    # RMS
    # ========================================================

    @staticmethod
    def rms(audio):

        if len(audio) == 0:

            return 0.0

        return float(
            np.sqrt(
                np.mean(
                    np.square(audio)
                )
            )
        )


    # ========================================================
    # RESAMPLE
    # ========================================================

    @staticmethod
    def resample_audio(
        audio,
        input_rate,
        output_rate
    ):

        if audio is None:

            return None


        if len(audio) == 0:

            return audio


        if input_rate == output_rate:

            return audio.astype(
                np.float32,
                copy=False
            )


        duration = (
            len(audio)
            / float(input_rate)
        )


        output_length = int(
            round(
                duration
                * output_rate
            )
        )


        if output_length <= 1:

            return audio.astype(
                np.float32,
                copy=False
            )


        old_positions = np.linspace(
            0,
            len(audio) - 1,
            num=len(audio)
        )


        new_positions = np.linspace(
            0,
            len(audio) - 1,
            num=output_length
        )


        resampled = np.interp(
            new_positions,
            old_positions,
            audio
        )


        return resampled.astype(
            np.float32
        )


    # ========================================================
    # RECORD SPEECH
    # ========================================================

    def record_until_silence(self):

        if self.sample_rate is None:

            return None


        pre_buffer = deque(
            maxlen=self.pre_buffer_blocks
        )


        recording = []

        speech_started = False

        recording_start = None

        silence_start = None


        print("--------------------------------")
        print("Listening...")
        print("--------------------------------")


        while True:

            try:

                data = self.audio_queue.get(
                    timeout=1.0
                )

            except queue.Empty:

                continue


            audio_block = np.asarray(
                data[:, 0],
                dtype=np.float32
            )


            rms = self.rms(
                audio_block
            )


            # =================================================
            # WAIT FOR SPEECH
            # =================================================

            if not speech_started:

                pre_buffer.append(
                    audio_block
                )


                if rms < MIN_RMS:

                    continue


                speech_started = True

                recording_start = (
                    time.monotonic()
                )

                silence_start = None


                recording.extend(
                    pre_buffer
                )

                pre_buffer.clear()


                print(
                    "Speech detected."
                )

                continue


            # =================================================
            # RECORDING
            # =================================================

            recording.append(
                audio_block
            )


            elapsed = (
                time.monotonic()
                - recording_start
            )


            # =================================================
            # SILENCE
            # =================================================

            if rms < SILENCE_RMS:

                if silence_start is None:

                    silence_start = (
                        time.monotonic()
                    )

            else:

                silence_start = None


            # =================================================
            # END AFTER SILENCE
            # =================================================

            if (

                silence_start is not None

                and

                (
                    time.monotonic()
                    - silence_start
                )

                >= SILENCE_DURATION

                and

                elapsed >= MIN_RECORDING_SECONDS

            ):

                break


            # =================================================
            # MAXIMUM RECORDING
            # =================================================

            if elapsed >= MAX_RECORDING_SECONDS:

                print(
                    "Maximum recording reached."
                )

                break


        if not recording:

            return None


        audio = np.concatenate(
            recording
        ).astype(
            np.float32,
            copy=False
        )


        # ====================================================
        # REMOVE DC OFFSET
        # ====================================================

        audio -= np.mean(
            audio
        )


        # ====================================================
        # ORIGINAL RMS
        # ====================================================

        original_rms = self.rms(
            audio
        )


        print(
            "Recorded RMS:",
            round(
                original_rms,
                5
            )
        )


        if original_rms < MIN_RMS:

            print(
                "Audio too quiet."
            )

            return None


        # ====================================================
        # NORMALIZE
        # ====================================================

        peak = float(
            np.max(
                np.abs(audio)
            )
        )


        if peak > 0.001:

            gain = min(
                0.9 / peak,
                3.0
            )

            audio *= np.float32(
                gain
            )


        # ====================================================
        # RESAMPLE TO 16 kHz
        # ====================================================

        print(
            "Resampling:",
            self.sample_rate,
            "Hz ->",
            WHISPER_SAMPLE_RATE,
            "Hz"
        )


        audio = self.resample_audio(

            audio,

            self.sample_rate,

            WHISPER_SAMPLE_RATE
        )


        print(
            "Recording complete."
        )


        return audio


# ============================================================
# WHISPER
# ============================================================

class SpeechRecognizer:

    def __init__(self):

        self.model = None


    # ========================================================
    # LOAD
    # ========================================================

    def load(self):

        print("--------------------------------")
        print("Loading Whisper...")
        print("--------------------------------")


        self.model = WhisperModel(

            WHISPER_MODEL,

            device=WHISPER_DEVICE,

            compute_type=WHISPER_COMPUTE_TYPE,

            cpu_threads=WHISPER_CPU_THREADS,

            num_workers=WHISPER_NUM_WORKERS
        )


        print("--------------------------------")
        print("Whisper loaded.")
        print("--------------------------------")


    # ========================================================
    # TRANSCRIBE
    # ========================================================

    def transcribe(
        self,
        audio
    ):

        if audio is None:

            return ""


        if self.model is None:

            return ""


        try:

            print("--------------------------------")
            print("Transcribing...")
            print("--------------------------------")


            segments, info = (

                self.model.transcribe(

                    audio,

                    language="en",

                    beam_size=5,

                    best_of=5,

                    temperature=0.0,

                    condition_on_previous_text=False,

                    vad_filter=True,

                    vad_parameters={

                        "min_silence_duration_ms": 400,

                        "speech_pad_ms": 200
                    },

                    no_speech_threshold=0.55,

                    log_prob_threshold=-1.0,

                    compression_ratio_threshold=2.0,

                    initial_prompt=(
                        "Sam. "
                        "Sam who is the prime minister of India? "
                        "Sam what is the weather? "
                        "Sam tell me the time."
                    )
                )
            )


            text_parts = []


            for segment in segments:

                text = segment.text.strip()


                if not text:

                    continue


                text_parts.append(
                    text
                )


            result = " ".join(
                text_parts
            ).strip()


            print("--------------------------------")
            print(
                "Recognized:",
                result
            )
            print("--------------------------------")


            return result


        except Exception as e:

            print("--------------------------------")
            print("WHISPER ERROR")
            print("--------------------------------")

            print(
                type(e).__name__,
                ":",
                e
            )

            print("--------------------------------")

            return ""


# ============================================================
# WAKE WORD
# ============================================================

class WakeWordDetector:

    def __init__(self):

        self.wake_word = WAKE_WORD


    # ========================================================
    # NORMALIZE
    # ========================================================

    @staticmethod
    def normalize(text):

        return (

            text.lower()

            .replace(",", " ")

            .replace(".", " ")

            .replace("?", " ")

            .replace("!", " ")

            .replace(";", " ")

            .replace(":", " ")

        )


    # ========================================================
    # SIMILARITY
    # ========================================================

    def similarity(
        self,
        word
    ):

        return difflib.SequenceMatcher(

            None,

            self.wake_word,

            word

        ).ratio()


    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        text
    ):

        normalized = self.normalize(
            text
        )


        words = normalized.split()


        # ====================================================
        # EXACT MATCH
        # ====================================================

        for word in words:

            if word in WAKE_VARIANTS:

                return True


        # ====================================================
        # FUZZY MATCH
        # ====================================================

        for word in words:

            if len(word) < 3:

                continue


            similarity = self.similarity(
                word
            )


            if similarity >= WAKE_SIMILARITY:

                print(
                    "Possible SAM match:",
                    word,
                    "similarity:",
                    round(
                        similarity,
                        2
                    )
                )

                return True


        return False


    # ========================================================
    # REMOVE WAKE WORD
    # ========================================================

    def remove_wake_word(
        self,
        text
    ):

        words = text.split()


        for index, word in enumerate(words):

            clean = (

                word

                .strip(
                    " ,.?;:!"
                )

                .lower()
            )


            # Exact

            if clean in WAKE_VARIANTS:

                words.pop(
                    index
                )

                return " ".join(
                    words
                ).strip()


            # Fuzzy

            if len(clean) >= 3:

                similarity = self.similarity(
                    clean
                )


                if similarity >= WAKE_SIMILARITY:

                    words.pop(
                        index
                    )

                    return " ".join(
                        words
                    ).strip()


        return text.strip()


# ============================================================
# AI MANAGER
# ============================================================

class AIManager:

    def __init__(self):

        self.gemini = None

        self.groq = None


    # ========================================================
    # INITIALIZE
    # ========================================================

    def initialize(self):

        if GEMINI_API_KEY:

            try:

                self.gemini = genai.Client(

                    api_key=GEMINI_API_KEY
                )

                print(
                    "Gemini initialized."
                )

            except Exception as e:

                print(
                    "Gemini initialization failed:",
                    e
                )

        else:

            print(
                "WARNING: GEMINI_API_KEY missing."
            )


        if GROQ_API_KEY:

            try:

                self.groq = Groq(

                    api_key=GROQ_API_KEY
                )

                print(
                    "Groq initialized."
                )

            except Exception as e:

                print(
                    "Groq initialization failed:",
                    e
                )

        else:

            print(
                "WARNING: GROQ_API_KEY missing."
            )


    # ========================================================
    # GEMINI
    # ========================================================

    def ask_gemini(
        self,
        command
    ):

        if self.gemini is None:

            return None


        response = (

            self.gemini.models.generate_content(

                model=GEMINI_MODEL,

                contents=(

                    SYSTEM_PROMPT

                    + "\n\nUser: "

                    + command

                    + "\n\nAssistant:"
                )
            )
        )


        answer = (

            response.text

            if response.text

            else ""
        ).strip()


        if not answer:

            return None


        return answer


    # ========================================================
    # GROQ
    # ========================================================

    def ask_groq(
        self,
        command
    ):

        if self.groq is None:

            return None


        response = (

            self.groq.chat.completions.create(

                model=GROQ_MODEL,

                messages=[

                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },

                    {
                        "role": "user",
                        "content": command
                    }

                ],

                temperature=0.2,

                max_tokens=100,

                reasoning_effort="none"
            )
        )


        if not response.choices:

            return None


        answer = (

            response
            .choices[0]
            .message
            .content
        )


        if not answer:

            return None


        return answer.strip()


    # ========================================================
    # PROCESS
    # ========================================================

    def process(
        self,
        command
    ):

        print("--------------------------------")
        print("COMMAND:")
        print(command)
        print("--------------------------------")


        # ====================================================
        # GEMINI FIRST
        # ====================================================

        try:

            answer = self.ask_gemini(
                command
            )


            if answer:

                print(
                    "Sam [Gemini]:",
                    answer
                )

                return answer


        except Exception as e:

            print(
                "Gemini failed:",
                type(e).__name__,
                e
            )


        # ====================================================
        # GROQ FALLBACK
        # ====================================================

        try:

            answer = self.ask_groq(
                command
            )


            if answer:

                print(
                    "Sam [Groq]:",
                    answer
                )

                return answer


        except Exception as e:

            print(
                "Groq failed:",
                type(e).__name__,
                e
            )


        return (
            "Sorry, I could not "
            "connect to the AI service."
        )


# ============================================================
# TEXT TO SPEECH
# ============================================================

class TTSManager:

    def __init__(
        self,
        audio_manager
    ):

        self.audio = audio_manager


    # ========================================================
    # GENERATE
    # ========================================================

    async def generate(
        self,
        text
    ):

        communicator = edge_tts.Communicate(

            text,

            TTS_VOICE,

            rate=TTS_RATE,

            volume=TTS_VOLUME
        )


        audio_data = bytearray()


        async for chunk in communicator.stream():

            if chunk["type"] == "audio":

                audio_data.extend(
                    chunk["data"]
                )


        return bytes(
            audio_data
        )


    # ========================================================
    # SPEAK
    # ========================================================

    def speak(
        self,
        text
    ):

        if not text:

            return


        print("--------------------------------")
        print("Speaking...")
        print("--------------------------------")


        # Stop microphone so Sam does not
        # hear its own voice.

        self.audio.stop()

        self.audio.clear_queue()


        try:

            audio_data = asyncio.run(

                self.generate(
                    text
                )
            )


            if not audio_data:

                return


            audio, sample_rate = (

                sf.read(

                    io.BytesIO(
                        audio_data
                    ),

                    dtype="float32"
                )
            )


            if audio.ndim > 1:

                audio = np.mean(
                    audio,
                    axis=1
                )


            sd.play(

                audio,

                samplerate=sample_rate,

                device=SPEAKER_DEVICE,

                blocking=True
            )


            sd.stop()


            print(
                "Finished speaking."
            )


        except Exception as e:

            print(
                "TTS ERROR:",
                type(e).__name__,
                e
            )


        finally:

            self.audio.clear_queue()

            time.sleep(
                0.3
            )


            try:

                self.audio.start()

                self.audio.clear_queue()

            except Exception as e:

                print(
                    "Could not restart microphone:",
                    e
                )


# ============================================================
# BASIC SAM ASSISTANT
# ============================================================

class SamAssistant:

    def __init__(self):

        self.audio = AudioManager()

        self.stt = SpeechRecognizer()

        self.wake = WakeWordDetector()

        self.ai = AIManager()

        self.tts = TTSManager(
            self.audio
        )


    # ========================================================
    # INITIALIZE
    # ========================================================

    def initialize(self):

        print("--------------------------------")
        print("Starting Sam Voice Assistant")
        print("--------------------------------")


        # ====================================================
        # AUDIO DEVICES
        # ====================================================

        self.audio.print_devices()


        # ====================================================
        # DETERMINE MICROPHONE RATE
        # ====================================================

        if not self.audio.determine_sample_rate():

            print(
                "Could not determine microphone."
            )

            return False


        # ====================================================
        # CHECK MICROPHONE
        # ====================================================

        if not self.audio.check_microphone():

            print(
                "Voice Assistant stopped."
            )

            return False


        # ====================================================
        # LOAD WHISPER
        # ====================================================

        try:

            self.stt.load()

        except Exception as e:

            print(
                "Whisper loading failed:",
                type(e).__name__,
                e
            )

            return False


        # ====================================================
        # AI
        # ====================================================

        self.ai.initialize()


        # ====================================================
        # START MICROPHONE
        # ====================================================

        try:

            self.audio.start()

        except Exception as e:

            print(
                "Microphone start failed:",
                type(e).__name__,
                e
            )

            return False


        print("--------------------------------")
        print("SAM VOICE ASSISTANT READY")
        print("--------------------------------")
        print("Say: SAM + your command")
        print("--------------------------------")


        return True


    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        if not self.initialize():

            return


        try:

            while True:

                # ============================================
                # RECORD
                # ============================================

                audio = (

                    self.audio
                    .record_until_silence()
                )


                if audio is None:

                    continue


                # ============================================
                # WHISPER
                # ============================================

                text = (

                    self.stt
                    .transcribe(audio)
                )


                if not text:

                    continue


                # ============================================
                # WAKE WORD
                # ============================================

                if not self.wake.detect(text):

                    print(
                        "Ignored - SAM not detected."
                    )

                    continue


                print("--------------------------------")
                print("SAM DETECTED")
                print("--------------------------------")


                # ============================================
                # REMOVE SAM
                # ============================================

                command = (

                    self.wake
                    .remove_wake_word(text)
                )


                if not command:

                    print(
                        "Waiting for command..."
                    )

                    continue


                print("--------------------------------")
                print(
                    "Command:",
                    command
                )
                print("--------------------------------")


                # ============================================
                # AI
                # ============================================

                answer = (

                    self.ai
                    .process(command)
                )


                # ============================================
                # SPEAK
                # ============================================

                self.tts.speak(
                    answer
                )


        except KeyboardInterrupt:

            print(
                "\nSam stopped."
            )


        except Exception as e:

            print(
                "Voice Assistant Error:",
                type(e).__name__,
                e
            )


        finally:

            self.audio.close()

            print("--------------------------------")
            print("Voice Assistant stopped.")
            print("--------------------------------")


# ============================================================
# FUNCTION USED BY main.py
# ============================================================

def start_voice_assistant():

    assistant = SamAssistant()

    assistant.run()


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    start_voice_assistant()
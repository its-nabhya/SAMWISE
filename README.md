# SAMWISE- Situationally Aware Multi-modal Wearable for Intelligence Safety & Emergency response

# Fall Detection — ML System

## Overview

The fall-detection subsystem is a **dual-IMU, two-branch 1D CNN system** running on the Arduino UNO Q Linux side with **ONNX Runtime**.

It uses:

- **Waist IMU:** W2 `BaselineCNN`, derived from a KFall-pretrained model and frozen for deployment.
- **Thigh IMU:** T1 `BaselineCNN`, trained on local thigh-IMU data.
- **Input:** 3-axis acceleration from each IMU.
- **Sampling target:** 100 Hz.
- **Window:** 2.5 s / 250 samples.
- **Stride:** 125 samples (50% overlap).
- **Inference:** CPU-based ONNX Runtime.

The gyroscope channels are retained in the sensor stream/logging pipeline but are **not used by the deployed CNNs**.

## ML Pipeline

```text
Synchronized Waist + Thigh IMU
              │
              ▼
      250-sample rolling windows
              │
       ┌──────┴──────┐
       ▼             ▼
   WAIST W2       THIGH T1
   BaselineCNN    BaselineCNN
       │             │
       ▼             ▼
  P(waist)        P(thigh)
       │             │
       └──────┬──────┘
              ▼
   P_fused = 0.85 P_waist
           + 0.15 P_thigh
              │
              ▼
       EMA smoothing
         α = 0.55
              │
              ▼
       Threshold = 0.52
              │
              ▼
        FALL / NORMAL
```

## 1. Waist Branch — W2 Transfer Learning

The waist branch uses a `BaselineCNN` initialized from the **KFall fall-detection dataset** through transfer learning.

KFall provides a substantially richer training basis than the local one-subject dataset, with:

- **5,075 recordings**
- **32 subjects**
- **21 ADL categories**
- **15 fall types**
- **2,729 ADL recordings**
- **2,346 fall recordings**
- 100 Hz inertial recordings with temporal fall annotations

The resulting W2 model is frozen for deployment.

### W2 deployment contract

```text
Live waist acceleration (m/s²)
        │
        ▼
Axis transform: [-X, -Y, +Z]
        │
        ▼
Convert m/s² → g
        │
        ▼
Frozen W2 normalization
        │
        ▼
Tensor: [1, 3, 250]
        │
        ▼
W2 ONNX
        │
        ▼
Fall logit → sigmoid → P_waist
```

W2 normalization:

```text
mean = [ 0.04393314, -0.42085612,  0.06521389]
std  = [ 0.17782952,  0.72337168,  0.54456365]
```

The deployed waist transformation is therefore:

```text
waist_ax → -waist_ax / 9.80665
waist_ay → -waist_ay / 9.80665
waist_az →  waist_az / 9.80665
```

followed by the frozen W2 channel-wise normalization.

## 2. Thigh Branch — T1

The thigh branch uses a locally trained 3-channel `BaselineCNN`.

Channel order:

```text
thigh_ax
thigh_ay
thigh_az
```

The deployed T1 preprocessing intentionally performs:

- **No axis transformation**
- **No sign flip**
- **No rotation**
- **No unit conversion**
- **No filtering**
- **No clipping**

It applies only the stored channel-wise normalization:

```text
ax' = (ax + 6.50183868) / 3.93414831
ay' = (ay + 1.05829346) / 10.27457523
az' = (az + 1.46983993) / 6.66970825
```

and forms:

```text
[1, 3, 250]
```

for ONNX inference.

T1 normalization:

```text
mean = [-6.50183868, -1.05829346, -1.46983993]
std  = [ 3.93414831, 10.27457523,  6.66970825]
```

### T1 calibration

A Platt calibration was explored during development, but the **deployed model does not apply Platt calibration**.

The deployed thigh probability is:

```text
P_thigh = sigmoid(T1_raw_logit)
```

The calibration parameters retained in the model metadata are historical/reference values only.

## 3. Probability Fusion

The deployed system uses the current fusion contract:

```text
P_fused = 0.85 × P_waist + 0.15 × P_thigh
```

The higher waist contribution preserves the information learned from the larger KFall transfer-learning source while retaining a complementary thigh signal.

## 4. Temporal Decision Layer

Each fused probability is passed through an exponential moving average:

```text
EMA_alpha = 0.55
```

The final fall decision is:

```text
FALL if EMA_probability >= 0.52
NORMAL otherwise
```

A latch prevents repeated `FALL DETECTED` messages from the same sustained above-threshold event. The latch resets once the decision returns below threshold.

## 5. Streaming and Windowing

The detector maintains independent rolling buffers for the waist and thigh accelerometers.

```text
Window length : 250 samples
Sampling      : 100 Hz
Window time   : 2.5 s
Stride        : 125 samples
Overlap       : 50%
```

The first inference is performed after the first complete 250-sample window. Subsequent inference is performed every 125 incoming samples.

Only accelerometer channels are passed to the ML models.

## 6. ONNX Runtime Deployment

The deployed models are:

```text
models/WAIST_W2_KF_FROZEN_CANONICAL_2.5s.onnx
models/T1-THIGH-A_2.5s.onnx
```

The UNO Q runs inference using:

```text
ONNX Runtime
CPUExecutionProvider
```

PyTorch is **not required on the deployed device**.

Each CNN produces a single fall logit:

```text
logit → sigmoid → probability
```

The application records:

- W2 logit/probability
- T1 raw logit/probability
- fused probability
- EMA probability
- final fall decision
- inference latency

in:

```text
stage3d6_ml_results.csv
```

## 7. Fall Detection Control Flow

```text
ESP32
  │
  │ synchronized IMU stream
  ▼
UNO Q main.py
  │
  ├──► ML queue ──► DualIMUFallDetector
  │                    │
  │                    ├── W2 ONNX
  │                    ├── T1 ONNX
  │                    ├── probability fusion
  │                    └── EMA + threshold
  │
  └──► FSM fallback
```

The threshold-based ML detector is the **primary fall detector**.

The legacy free-fall/impact/stillness FSM is retained only as a **fallback**. It is enabled when the ML worker is unavailable; it does not operate as a parallel primary detector while ML inference is healthy.

When ML detects a fall, `main.py` sends:

```text
FALL DETECTED
```

to the ESP32.

The existing notification system can then use the fall event and GPS information for emergency alerts.

## 8. Real-Time PoC Results

Real-time testing on the Arduino UNO Q with live IMU data showed:

- Approximately **89% of observed falls detected**.
- Tested ADLs were consistently distinguished, including:
  - walking
  - sitting/getting up
  - bending
  - shoelace-related movements
  - jogging
  - lying down/getting up
- Main known false-positive case:
  - continuous jumping
  - jumping from height
- These high-impact activities were misclassified as falls in approximately **60% of tested cases**.

These figures are **PoC observations**, not population-level accuracy estimates.

## 9. Known Failure Modes

Performance can degrade when:

- high-impact non-fall motion resembles a fall;
- sensor mounting orientation differs from the deployment contract;
- sensor units or axis conventions change;
- samples are missing or timing becomes irregular;
- movements are substantially different from training data;
- a fall type is poorly represented in the training data.

The current jumping failure mode is the primary target for future improvement.

## 10. Future ML Improvements

### Lightweight domain adaptation

A potential next step is a **linear adapter layer** placed around the transferred waist representation. Its purpose would be to provide lightweight adaptation to the local sensor/user domain without replacing the complete pretrained model.

### Reinforcement-learning-based personalization

The longer-term direction is **user-specific personalization using reinforcement learning**:

```text
Base pretrained model
        │
        ▼
User-specific feedback
        +
Hard-negative examples
        │
        ▼
Lightweight adaptive layer / decision policy
        │
        ▼
Personalized fall detector
```

The goal is to improve accuracy for an individual user's movement patterns while retaining the pretrained model as the initial knowledge base.

---


# SanWise Voice Assistant
## Overview
The SAMWISE Voice Assistant is a modular voice-controlled AI assistant implemented in `voice_assistant.py` and integrated with the main application through `main.py`.
The assistant continuously captures audio from a USB microphone, detects speech, converts it into text, checks for the **SAM** wake word, processes the user's command using an AI model, and converts the response back into speech.
The voice assistant is started from `main.py` using:
```python
from voice_assistant import start_voice_assistant
start_voice_assistant()
```
# System Pipeline
```text
USB Microphone
      │
      ▼
sounddevice InputStream
      │
      ▼
Audio Callback
      │
      ▼
Audio Queue
      │
      ▼
RMS-Based Speech Detection
      │
      ▼
Pre-buffer + Speech Recording
      │
      ▼
Silence Detection
      │
      ▼
Audio Processing
      │
      ▼
Resample to 16 kHz
      │
      ▼
Faster-Whisper (base.en)
      │
      ▼
Speech-to-Text
      │
      ▼
SAM Wake Word Detection
      │
 ┌────┴─────┐
 │          │
NO         YES
 │          │
 ▼          ▼
Ignore   Remove Wake Word
             │
             ▼
        User Command
             │
             ▼
Gemini API (gemini-3.6-flash)
             │
      ┌──────┴──────┐
      │             │
   Success        Failure
      │             │
      │             ▼
      │     Groq API (qwen/qwen3.6-27b)
      │             │
      └──────┬──────┘
             │
             ▼
        AI Response
             │
             ▼
Edge TTS (en-US-GuyNeural)
             │
             ▼
          Speaker
             │
             ▼
       Return to Listening
```
# Working of the Voice Assistant
## 1. Initialization
The voice assistant starts when `main.py` calls:
```python
from voice_assistant import start_voice_assistant
start_voice_assistant()
```
This creates a `SamAssistant` object and starts the main execution loop.
During initialization, the system Lists available audio devices then Detects the USB microphone. then Determines the microphone's native sample rate.Checks the microphone configuration.Loads the Faster-Whisper model.Initializes Gemini and Groq clients.Starts the microphone audio stream.
The main controller connects the following modules:

## 2. Audio Capture and Speech Detection

AudioManager uses sounddevice.InputStream to continuously capture audio from the USB microphone.
Audio blocks are placed into a queue and analyzed using RMS (Root Mean Square) to distinguish speech from silence or background noise.
When the RMS level exceeds MIN_RMS, speech recording begins. 
A 0.30-second pre-buffer helps preserve the beginning of the user's speech.Recording stops when approximately 0.9 seconds of silence is detected or the 8-second maximum recording time is reached.

## 3. Audio Processing

The recorded audio is prepared for speech recognition by:
Removing DC offset.
Checking whether the recording is sufficiently loud.
Normalizing the audio amplitude.
Resampling it from the microphone's native sample rate to 16 kHz.
The resulting audio is then ready for Faster-Whisper.

## 4. Speech-to-Text Using Faster-Whisper
Faster-Whisper converts the processed audio into text using:

WHISPER_MODEL = "base.en"

It runs on the CPU with INT8 computation to reduce resource usage.

For example:

Speech:       "Sam, what is the weather?"
Recognized:   "Sam what is the weather?"

## 5. SAM Wake Word Detection
The recognized text is checked for the SAM wake word.
Two methods are used:
Exact matching with variants such as sam, samm, and saam.
Fuzzy matching using difflib.SequenceMatcher to handle minor speech-recognition errors.
If SAM is not detected, the input is ignored.
When SAM is detected, it is removed from the recognized text and the remaining text becomes the command.
"Sam what is the weather?"
          ↓
"What is the weather?"

## 6. AI Processing
The extracted command is first sent to Gemini:
GEMINI_MODEL = "gemini-3.6-flash"
Gemini is the primary AI model responsible for generating the response.
If Gemini fails or does not return a valid response, the system uses Qwen through Groq as a fallback:
GROQ_MODEL = "qwen/qwen3.6-27b"
If both services fail, the assistant returns an error message.

## 7. Text-to-Speech
The AI-generated response is converted into speech using Edge TTS:
TTS_VOICE = "en-US-GuyNeural"
Before speaking, the microphone is temporarily stopped and its queue is cleared. This prevents SAM from detecting its own response.
The generated audio is played through the speaker. After playback, the microphone is restarted and the assistant continues listening.

# Module Responsibilities
| Module             | Responsibility                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| `AudioManager`     | Captures microphone audio, manages the audio queue, detects speech and silence, and preprocesses audio. |
| `SpeechRecognizer` | Uses Faster-Whisper (`base.en`) to convert speech into text.                                            |
| `WakeWordDetector` | Detects the SAM wake word using exact and fuzzy matching and extracts the command.                      |
| `AIManager`        | Uses Gemini (`gemini-3.6-flash`) as the primary model and  Groq (`qwen/qwen3.6-27b`) as a fallback.     |
| `TTSManager`       | Converts the AI response into speech using Edge TTS (`en-US-GuyNeural`).                                |
| `SamAssistant`     | Connects and controls the complete voice assistant pipeline.                                            |

# Technology and Model Summary

| Function          | Technology / Model      |
| ----------------- | ----------------------- |
| Audio Capture     | `sounddevice`           |
| Audio Processing  | `NumPy`                 |
| Speech-to-Text    | Faster-Whisper          |
| Whisper Model     | `base.en`               |
| Primary AI        | Gemini                  |
| Gemini Model      | `gemini-3.6-flash`      |
| Fallback AI Model | Groq `qwen/qwen3.6-27b` |
| Text-to-Speech    | Edge TTS                |
| TTS Voice         | `en-US-GuyNeural`       |

# SamWise Home Automation System
## Overview
The Blynk module provides remote control of three physical LEDs and displays the current weather information through the Blynk app and the UNO Q LED Matrix.
The system is divided into two parts:
- `blynk.py` — Handles Blynk communication, virtual pins, LED commands, and weather data on the Linux side of the UNO Q.
- `sketch.ino` — Handles the physical LEDs, LED Matrix, and Arduino Bridge functions on the microcontroller side.
Both modules are started/used by `main.py`.

# System Pipeline
```text
                         main.py
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
          blynk.py                   Weather System
              │                           │
              ▼                           ▼
       Blynk Cloud/App              Weather Forecast
              │                           │
       V0 ──► LED1                     Weather
       V1 ──► LED2                         │
       V2 ──► LED3                         ▼
              │                     blynk.py stores
              │                     latest weather
              ▼                           │
       Arduino Bridge ◄───────────────────┘
              │
              ▼
         sketch.ino
              │
       ┌──────┴────────┐
       ▼               ▼
   D4/D5/D6        LED Matrix
       │               │
       ▼               ▼
    LED1-3       Weather Animation
```
Structure and Working
1. blynk.py — Blynk Communication
blynk.py runs Blynk communication in a background thread, allowing the main application to continue running independently.
It connects to Blynk Cloud and maps the virtual pins:
Virtual Pin	Function
V0	LED1 control
V1	LED2 control
V2	LED3 control
V3	Weather category
V4	Weather description

When a user changes an LED switch in the Blynk app, the corresponding callback is triggered. The requested state is stored and sent to sketch.ino through Bridge.call().
```text
Blynk V0/V1/V2
      ↓
Callback Handler
      ↓
set_led1 / set_led2 / set_led3
      ↓
Arduino Bridge
```
A threading.Lock protects shared LED and weather states from simultaneous access by different threads.

2. Weather Data Handling
main.py provides updated weather information to blynk.py using:
update_weather_dashboard(category, description)
The latest weather category and description are stored safely using the shared state lock.
blynk.py periodically sends these values to:
V3 → Weather category
V4 → Weather description
Weather data is also resent after a Blynk reconnection so that the dashboard remains synchronized.

3. sketch.ino — Hardware Control
sketch.ino runs on the Arduino side and controls the physical hardware.
The three LEDs are connected to:
LED1 → D4
LED2 → D5
LED3 → D6
Bridge.provide() exposes the LED control functions to Python:
Bridge.provide("set_led1", set_led1);
Bridge.provide("set_led2", set_led2);
Bridge.provide("set_led3", set_led3);
Therefore, when blynk.py calls:
Bridge.call("set_led1", state)
the corresponding Arduino function changes the physical LED state.

4. Weather LED Matrix
The Arduino sketch also controls the UNO Q LED Matrix.
It requests the current weather category through:
Bridge.call("get_weather_forecast")

Depending on the returned category, the appropriate animation is displayed:
Weather	Matrix Animation
Sunny	sunny
Cloudy	cloudy
Rainy	rainy
Snowy	snowy
Foggy	foggy
The animation is played repeatedly using playRepeat().

5. Reliability
The Blynk connection runs continuously in a daemon background thread.
If the connection fails:
The Blynk object is cleared.
The system waits 5 seconds.
A new connection attempt is made.
Current LED states and weather information are sent again after reconnection.

This allows the Blynk functionality to recover without stopping the main application.

# SamWise Emergency Notification System
## Overview
The SAMWISE notification system sends an emergency alert when a fall is detected. It uses two independent notification modules:
- `email_sender.py` — Sends the alert through Gmail SMTP.
- `telegram_sender.py` — Sends the alert to multiple Telegram users through the Telegram Bot API.
Both modules are called by `main.py` when a fall is detected.
# System Pipeline
```text
                    Fall Detected
                         │
                         ▼
                       main.py
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      email_sender.py       telegram_sender.py
              │                     │
              ▼                     ▼
        Gmail SMTP            Telegram Bot API
              │                     │
              ▼                     ▼
        Email Alert          Telegram Alert
              │                     │
              └──────────┬──────────┘
                         ▼
                  Time + Location
                         │
                         ▼
                    Google Maps
```                    
Structure and Working
1. main.py — Alert Trigger
When the fall detection logic in main.py identifies a fall, it provides the relevant information to both notification functions:
send_email(
    fall_time,
    latitude,
    longitude
)
send_telegram(
    fall_time,
    latitude,
    longitude
)
The notification modules do not perform fall detection themselves. They are responsible only for delivering the alert.
2. email_sender.py — Email Notification
send_email() creates an emergency email containing:
Fall detection time
Latitude
Longitude
Google Maps location link

The module uses Python's smtplib to connect to Gmail's SMTP server:
Gmail SMTP
smtp.gmail.com:587

The connection uses TLS encryption and authenticates using the configured Gmail account and app password.
The email is then sent to all addresses listed in RECIPIENTS.

3. telegram_sender.py — Telegram Notification
send_telegram() creates a similar emergency message containing the fall time and GPS coordinates.
The message is sent using the Telegram Bot API through an HTTP POST request.
The module loops through all configured CHAT_IDS, allowing the same alert to be delivered to multiple people.
Each request has a 10-second timeout and its result is checked to determine whether the message was successfully delivered.

4. Location Information

Both notification methods include the detected GPS coordinates:

Latitude
Longitude

A Google Maps URL is automatically generated using these coordinates:

https://maps.google.com/?q=latitude,longitude

This allows the recipient to directly open the detected location on Google Maps.

5. Error Handling

The two notification systems operate independently.
If Telegram delivery fails, the email notification can still be sent, and vice versa.
telegram_sender.py also checks each recipient individually and returns False if any message fails.
email_sender.py returns True after successfully completing the email transmission.

# Module Responsibilities
| Module              | Responsibility                                                                |
| ------------------- | ----------------------------------------------------------------------------- |
| `main.py`           | Detects the fall and triggers the email and Telegram notification modules.    |
| `email_sender.py`   | Creates and sends emergency fall alerts using Gmail SMTP.                     |
| `telegram_sender.py`| Sends emergency fall alerts to multiple users through the Telegram Bot API.   |
| `GPS Data`          | Provides the fall time, latitude, and longitude for the emergency alert.      |

# Technology and Model Summary
| Function             | Technology / Model                                      |
| -------------------- | ------------------------------------------------------- |
| Email                | Python `smtplib`                                        |
| Email Security       | Gmail SMTP + TLS                                        |
| Telegram             | Telegram Bot API                                        |
| HTTP Communication   | `requests`                                              |
| Location             | GPS Latitude + Longitude                                |
| Map Link             | Google Maps                                             |
| Trigger              | Fall detection from `main.py`                           |

The result is a dual-channel emergency notification system, providing both email and Telegram alerts with the detected fall time and location.
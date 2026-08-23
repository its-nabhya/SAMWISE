import time
import socket
import threading
import os
import csv

from pathlib import Path
from queue import Queue
from collections import deque

from math import radians
from math import sin
from math import cos
from math import sqrt
from math import atan2

from datetime import datetime
from datetime import UTC
from datetime import timedelta

from arduino.app_utils import App
from arduino.app_utils import Bridge
from arduino.app_bricks.weather_forecast import WeatherForecast

from fall_ml import DualIMUFallDetector

from email_sender import send_email
from telegram_sender import send_telegram
from blynk import start_blynk
from blynk import update_weather_dashboard

from voice_assistant import start_voice_assistant


# ==========================================================
# TIME ZONE
# ==========================================================

IST = timedelta(hours=5, minutes=30)


# ==========================================================
# STAGE 3D.6 DUAL-IMU ML FALL DETECTOR
# ==========================================================

WAIST_MODEL_PATH = (
    "models/WAIST_W2_KF_FROZEN_CANONICAL_2.5s.onnx"
)

THIGH_MODEL_PATH = (
    "models/T1-THIGH-A_2.5s.onnx"
)

fall_ml = DualIMUFallDetector(
    waist_model_path=WAIST_MODEL_PATH,
    thigh_model_path=THIGH_MODEL_PATH,
    verbose=True,
)


# ==========================================================
# ML SETTINGS
# ==========================================================

# ML probability above this value means FALL.
ML_FALL_THRESHOLD = 0.52


# ==========================================================
# ML SHARED STATE
# ==========================================================

# Latest EMA probability produced by ML.

ml_probability = 0.0


# Latest ML fall decision.

ml_fall_decision = False


# Indicates whether ML is currently working correctly.

ml_available = False


# Protects the shared ML state.

ml_state_lock = threading.Lock()


# Prevent repeated FALL DETECTED messages from the same
# ML inference window.

ml_fall_latched = False


# Incoming accelerometer samples.

ml_queue = Queue()


# ==========================================================
# STAGE 3D.6 LOG
# ==========================================================

STAGE3D6_LOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "stage3d6_ml_results.csv"
)

STAGE3D6_LOG_FIELDS = [
    "timestamp_utc",
    "window_index",
    "w2_logit",
    "w2_probability",
    "t1_raw_logit",
    "t1_raw_probability",
    "t1_calibrated_probability",
    "fusion_probability",
    "ema_probability",
    "fall_decision",
    "inference_latency_ms",
]


def log_stage3d6_result(result):

    file_exists = (
        STAGE3D6_LOG_PATH.exists()
        and
        STAGE3D6_LOG_PATH.stat().st_size > 0
    )

    with STAGE3D6_LOG_PATH.open(
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=STAGE3D6_LOG_FIELDS
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow({

            "timestamp_utc":
                datetime.now(UTC).isoformat(),

            "window_index":
                result.window_index,

            "w2_logit":
                f"{result.w2_logit:.9f}",

            "w2_probability":
                f"{result.w2_probability:.9f}",

            "t1_raw_logit":
                f"{result.t1_raw_logit:.9f}",

            "t1_raw_probability":
                f"{result.t1_raw_probability:.9f}",

            "t1_calibrated_probability":
                f"{result.t1_calibrated_probability:.12f}",

            "fusion_probability":
                f"{result.fusion_probability:.9f}",

            "ema_probability":
                f"{result.ema_probability:.9f}",

            "fall_decision":
                int(result.fall_decision),

            "inference_latency_ms":
                f"{result.inference_latency_ms:.6f}",
        })


# ==========================================================
# ESP32 CONNECTION
# ==========================================================

ESP32_IP = "10.188.197.212"

PORT = 5000

sock = None

sock_lock = threading.Lock()

esp32_thread_running = True


# ==========================================================
# CONNECT TO ESP32
# ==========================================================

def connect_to_esp32():

    global sock

    while esp32_thread_running:

        new_sock = None

        try:

            print("--------------------------------")
            print("Connecting to ESP32...")
            print("--------------------------------")

            new_sock = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM
            )

            new_sock.settimeout(5)

            new_sock.connect(
                (ESP32_IP, PORT)
            )

            new_sock.settimeout(1)

            with sock_lock:

                if sock is not None:

                    try:
                        sock.close()

                    except OSError:
                        pass

                sock = new_sock

            print("--------------------------------")
            print("Connected to ESP32")
            print("--------------------------------")

            return new_sock

        except Exception as e:

            print("--------------------------------")
            print("ESP32 CONNECTION FAILED")
            print("Retrying in 2 seconds...")
            print("Error:", e)
            print("--------------------------------")

            if new_sock is not None:

                try:
                    new_sock.close()

                except OSError:
                    pass

            time.sleep(2)

    return None


# ==========================================================
# CLOSE ESP32 SOCKET
# ==========================================================

def close_esp32_socket():

    global sock

    with sock_lock:

        old_sock = sock

        sock = None

    if old_sock is not None:

        try:

            old_sock.shutdown(
                socket.SHUT_RDWR
            )

        except OSError:
            pass

        try:

            old_sock.close()

        except OSError:
            pass


# ==========================================================
# SEND DATA TO ESP32
# ==========================================================

def send_to_esp32(message):

    global sock

    try:

        with sock_lock:

            current_sock = sock

            if current_sock is None:

                print("--------------------------------")
                print("ESP32 NOT CONNECTED")
                print(
                    "Message not sent:",
                    message
                )
                print("--------------------------------")

                return False

            current_sock.sendall(
                (
                    message + "\n"
                ).encode()
            )

        print("--------------------------------")
        print(
            "Sent to ESP32:",
            message
        )
        print("--------------------------------")

        return True

    except (
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
        OSError
    ) as e:

        print("--------------------------------")
        print("ESP32 SEND ERROR")
        print("Error:", e)
        print("Connection will be re-established.")
        print("--------------------------------")

        close_esp32_socket()

        return False


# ==========================================================
# WEATHER
# ==========================================================

forecaster = WeatherForecast()


# ==========================================================
# FALL DETECTION FSM
#
# IMPORTANT:
#
# This FSM is now ONLY A FALLBACK.
#
# It is used only when:
#
#     ml_available == False
#
# When ML is healthy, the FSM does not send FALL DETECTED.
# ==========================================================

NORMAL = 0
FREE_FALL = 1
IMPACT = 2

fall_state = NORMAL

free_fall_time = 0.0
impact_time = 0.0

FREE_FALL_THRESHOLD = 4.0

IMPACT_THRESHOLD1 = 10.0
IMPACT_THRESHOLD2 = 6.0

STILL_THRESHOLD = 1.5

IMPACT_WINDOW = 1.0

STILL_TIME = 1.5


# ==========================================================
# GPS
# ==========================================================

gps_status = 0

current_lat = 22.494510
current_lon = 88.325614

last_weather_lat = current_lat
last_weather_lon = current_lon

last_weather_update = 0

weather_code = -1
weather_description = "UNKNOWN"
weather_category = "UNKNOWN"


# ==========================================================
# WEATHER LOCK
# ==========================================================

weather_lock = threading.Lock()


# ==========================================================
# FALL DETECTION FSM
# ==========================================================

def process_fall_detection(
    waist_ax,
    waist_ay,
    waist_az,
    thigh_ax,
    thigh_ay,
    thigh_az
):

    global fall_state
    global free_fall_time
    global impact_time

    # ------------------------------------------------------
    # DO NOT USE FSM WHEN ML IS WORKING
    # ------------------------------------------------------

    with ml_state_lock:

        ml_ok = ml_available

    if ml_ok:

        return

    # ------------------------------------------------------
    # ML FAILED
    # FSM IS NOW THE FALLBACK
    # ------------------------------------------------------

    now = time.time()

    waist_acc = sqrt(
        waist_ax * waist_ax +
        waist_ay * waist_ay +
        waist_az * waist_az
    )

    thigh_acc = sqrt(
        thigh_ax * thigh_ax +
        thigh_ay * thigh_ay +
        thigh_az * thigh_az
    )

    waist_move = abs(
        waist_acc - 9.81
    )

    thigh_move = abs(
        thigh_acc - 9.81
    )

    # ======================================================
    # NORMAL
    # ======================================================

    if fall_state == NORMAL:

        if (
            waist_acc < FREE_FALL_THRESHOLD
            and
            thigh_acc < FREE_FALL_THRESHOLD
        ):

            fall_state = FREE_FALL

            free_fall_time = now

            print("--------------------------------")
            print("FSM FALLBACK: FREE FALL")
            print("--------------------------------")

    # ======================================================
    # FREE FALL
    # ======================================================

    elif fall_state == FREE_FALL:

        if (
            now - free_fall_time
            > IMPACT_WINDOW
        ):

            fall_state = NORMAL

        elif (
            waist_acc > IMPACT_THRESHOLD1
            and
            thigh_acc > IMPACT_THRESHOLD2
        ):

            fall_state = IMPACT

            impact_time = now

            print("--------------------------------")
            print("FSM FALLBACK: IMPACT")
            print("--------------------------------")

    # ======================================================
    # IMPACT
    # ======================================================

    elif fall_state == IMPACT:

        if (
            now - impact_time > STILL_TIME
            and
            waist_move < STILL_THRESHOLD
            and
            thigh_move < STILL_THRESHOLD
        ):

            print("--------------------------------")
            print("FSM FALLBACK: FALL DETECTED")
            print("--------------------------------")

            if send_to_esp32(
                "FALL DETECTED"
            ):

                print(
                    "FSM FALL DETECTED sent to ESP32"
                )

            else:

                print(
                    "ESP32 unavailable - "
                    "FSM FALL DETECTED could not "
                    "be sent."
                )

            fall_state = NORMAL

        elif (
            now - impact_time > 5.0
        ):

            fall_state = NORMAL


# ==========================================================
# CSV HEADERS
# ==========================================================

ACCTEST_HEADER = (
    "timestampUNOQ,"
    "waist_ax,"
    "waist_ay,"
    "waist_az,"
    "waist_gx,"
    "waist_gy,"
    "waist_gz,"
    "thigh_ax,"
    "thigh_ay,"
    "thigh_az,"
    "thigh_gx,"
    "thigh_gy,"
    "thigh_gz\n"
)


GPS_HEADER = (
    "timestamp,"
    "gps_status,"
    "latitude,"
    "longitude,"
    "weather_code,"
    "weather_description,"
    "weather_category\n"
)


FALL_HEADER = (
    "timestampUNOQ,"
    "timestamp,"
    "label\n"
)


FALLDATA_HEADER = ACCTEST_HEADER


# ==========================================================
# CREATE CSV
# ==========================================================

def create_csv_if_needed(
    filename,
    header
):

    if (
        not os.path.exists(filename)
        or
        os.path.getsize(filename) == 0
    ):

        with open(
            filename,
            "w"
        ) as f:

            f.write(header)


create_csv_if_needed(
    "fall.csv",
    FALL_HEADER
)

create_csv_if_needed(
    "gps.csv",
    GPS_HEADER
)

create_csv_if_needed(
    "falldata.csv",
    FALLDATA_HEADER
)

create_csv_if_needed(
    "acctest.csv",
    ACCTEST_HEADER
)


# ==========================================================
# ESP32 RECEIVE BUFFER
# ==========================================================

esp32_buffer = ""


# ==========================================================
# IMU BUFFER
# ==========================================================

imu_buffer = deque()

ACCTEST_UPDATE_INTERVAL = 60

IMU_BUFFER_DURATION = 65

last_acctest_update = time.time()


# ==========================================================
# STORE IMU SAMPLE
# ==========================================================

def store_imu_sample(
    timestamp,
    imu_line
):

    global last_acctest_update

    imu_buffer.append(
        (
            timestamp,
            imu_line
        )
    )

    cutoff = (
        timestamp
        -
        IMU_BUFFER_DURATION
    )

    while (
        imu_buffer
        and
        imu_buffer[0][0] < cutoff
    ):

        imu_buffer.popleft()

    if (
        timestamp
        -
        last_acctest_update
        >= ACCTEST_UPDATE_INTERVAL
    ):

        write_acctest_csv(
            timestamp
        )

        last_acctest_update = timestamp


# ==========================================================
# WRITE ACCTEST
# ==========================================================

def write_acctest_csv(
    current_timestamp
):

    cutoff = (
        current_timestamp
        -
        60
    )

    try:

        with open(
            "acctest.csv",
            "w"
        ) as f:

            f.write(
                ACCTEST_HEADER
            )

            for (
                sample_timestamp,
                imu_line
            ) in imu_buffer:

                if (
                    sample_timestamp
                    >= cutoff
                ):

                    f.write(
                        f"{sample_timestamp},"
                        f"{imu_line}\n"
                    )

        print("--------------------------------")
        print("acctest.csv UPDATED")
        print(
            "Stored IMU data from:",
            cutoff,
            "to:",
            current_timestamp
        )
        print("--------------------------------")

    except Exception as e:

        print(
            "acctest.csv update error:",
            e
        )


# ==========================================================
# SAVE 20 SECONDS BEFORE FALL
# ==========================================================

def save_fall_data(
    fall_timestamp
):

    cutoff = (
        fall_timestamp
        -
        20
    )

    samples_saved = 0

    try:

        with open(
            "falldata.csv",
            "a"
        ) as f:

            for (
                sample_timestamp,
                imu_line
            ) in imu_buffer:

                if (
                    cutoff
                    <= sample_timestamp
                    <= fall_timestamp
                ):

                    f.write(
                        f"{sample_timestamp},"
                        f"{imu_line}\n"
                    )

                    samples_saved += 1

        print("--------------------------------")
        print("falldata.csv UPDATED")
        print(
            "Fall timestamp :",
            fall_timestamp
        )
        print(
            "Data window    :",
            cutoff,
            "to",
            fall_timestamp
        )
        print(
            "Samples saved  :",
            samples_saved
        )
        print("--------------------------------")

    except Exception as e:

        print(
            "falldata.csv update error:",
            e
        )


# ==========================================================
# GPS DISTANCE
# ==========================================================

def distance_meters(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371000.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        +
        cos(lat1)
        *
        cos(lat2)
        *
        sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c


# ==========================================================
# WEATHER UPDATE
# ==========================================================

def update_weather():

    global weather_code
    global weather_description
    global weather_category

    global last_weather_update
    global last_weather_lat
    global last_weather_lon

    global current_lat
    global current_lon

    try:

        print("--------------------------------")
        print("Updating Weather...")
        print("--------------------------------")

        forecast = (
            forecaster
            .get_forecast_by_coords(
                latitude=str(
                    current_lat
                ),
                longitude=str(
                    current_lon
                )
            )
        )

        with weather_lock:

            weather_code = (
                forecast.code
            )

            weather_description = (
                forecast.description
            )

            weather_category = (
                forecast.category
            )

            last_weather_update = (
                time.time()
            )

            last_weather_lat = (
                current_lat
            )

            last_weather_lon = (
                current_lon
            )

        update_weather_dashboard(
            weather_category,
            weather_description
        )

        print("--------------------------------")
        print("Weather Updated")
        print(
            "Code        :",
            weather_code
        )
        print(
            "Description :",
            weather_description
        )
        print(
            "Category    :",
            weather_category
        )
        print("--------------------------------")

    except Exception as e:

        print(
            "Weather Update Failed:",
            e
        )


# ==========================================================
# WEATHER THREAD
# ==========================================================

def weather_thread():

    global current_lat
    global current_lon

    while True:

        try:

            refresh = False

            if (
                time.time()
                -
                last_weather_update
                >= 60
            ):

                refresh = True

            else:

                moved = distance_meters(
                    last_weather_lat,
                    last_weather_lon,
                    current_lat,
                    current_lon
                )

                if moved >= 100:

                    refresh = True

            if refresh:

                update_weather()

        except Exception as e:

            print(
                "Weather Thread Error:",
                e
            )

        time.sleep(5)


# ==========================================================
# BRIDGE FUNCTIONS
# ==========================================================

def get_weather_forecast():

    with weather_lock:

        return weather_category


def get_weather_description():

    with weather_lock:

        return weather_description


def get_weather_code():

    with weather_lock:

        return weather_code


Bridge.provide(
    "get_weather_forecast",
    get_weather_forecast
)

Bridge.provide(
    "get_weather_description",
    get_weather_description
)

Bridge.provide(
    "get_weather_code",
    get_weather_code
)


# ==========================================================
# PROCESS GPS
# ==========================================================

def process_gps_message(
    line
):

    global current_lat
    global current_lon
    global gps_status

    try:

        parts = line.split(",")

        lat = float(parts[1])
        lon = float(parts[2])

        with weather_lock:

            current_lat = lat
            current_lon = lon

            if (
                lat == 22.494510
                and
                lon == 88.325614
            ):

                gps_status = 0

            else:

                gps_status = 1

        print("--------------------------------")
        print("GPS DATA")
        print(
            "Latitude :",
            current_lat
        )
        print(
            "Longitude:",
            current_lon
        )
        print(
            "Weather  :",
            weather_category
        )
        print("--------------------------------")

        with open(
            "gps.csv",
            "a"
        ) as f:

            f.write(
                f"{time.time()},"
                f"{gps_status},"
                f"{current_lat},"
                f"{current_lon},"
                f"{weather_code},"
                f"{weather_description},"
                f"{weather_category}\n"
            )

    except Exception as e:

        print(
            "GPS Parse Error:",
            e
        )


# ==========================================================
# PROCESS FALL TRUE
# ==========================================================

def process_fall_true():

    print("--------------------------------")
    print("CONFIRMED FALL")
    print("--------------------------------")

    fall_timestamp_epoch = time.time()

    fall_time = (
        datetime.now(UTC)
        +
        IST
    ).strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    try:

        with open(
            "fall.csv",
            "a"
        ) as f:

            f.write(
                f"{fall_timestamp_epoch},"
                f"{fall_time},"
                f"TRUE\n"
            )

    except Exception as e:

        print(
            "fall.csv write error:",
            e
        )

    save_fall_data(
        fall_timestamp_epoch
    )

    try:

        send_email(
            fall_time=fall_time,
            latitude=current_lat,
            longitude=current_lon
        )

    except Exception as e:

        print(
            "Email notification error:",
            e
        )

    try:

        send_telegram(
            fall_time=fall_time,
            latitude=current_lat,
            longitude=current_lon
        )

    except Exception as e:

        print(
            "Telegram notification error:",
            e
        )


# ==========================================================
# PROCESS FALL FALSE
# ==========================================================

def process_fall_false():

    print("--------------------------------")
    print("FALSE ALARM")
    print("--------------------------------")

    fall_timestamp_epoch = time.time()

    fall_time = (
        datetime.now(UTC)
        +
        IST
    ).strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    try:

        with open(
            "fall.csv",
            "a"
        ) as f:

            f.write(
                f"{fall_timestamp_epoch},"
                f"{fall_time},"
                f"FALSE\n"
            )

    except Exception as e:

        print(
            "fall.csv write error:",
            e
        )

    save_fall_data(
        fall_timestamp_epoch
    )


# ==========================================================
# PROCESS IMU
# ==========================================================

def process_imu_message(
    line
):

    print(line)

    imu_timestamp = time.time()

    store_imu_sample(
        imu_timestamp,
        line
    )

    try:

        (
            waist_ax,
            waist_ay,
            waist_az,
            waist_gx,
            waist_gy,
            waist_gz,
            thigh_ax,
            thigh_ay,
            thigh_az,
            thigh_gx,
            thigh_gy,
            thigh_gz
        ) = map(
            float,
            line.split(",")
        )

        # ==================================================
        # SEND ONLY ACCELEROMETER DATA TO ML QUEUE
        # ==================================================

        ml_queue.put((
            waist_ax,
            waist_ay,
            waist_az,
            thigh_ax,
            thigh_ay,
            thigh_az,
        ))

        # ==================================================
        # FSM FALLBACK
        #
        # This function immediately returns if ML is healthy.
        #
        # Therefore FSM processing does NOT interfere with ML.
        # ==================================================

        process_fall_detection(
            waist_ax,
            waist_ay,
            waist_az,
            thigh_ax,
            thigh_ay,
            thigh_az
        )

    except Exception as e:

        print(
            "IMU Parse Error:",
            e
        )


# ==========================================================
# ML WORKER THREAD
# ==========================================================

def ml_worker_thread():

    global ml_probability
    global ml_fall_decision
    global ml_available
    global ml_fall_latched

    while True:

        sample = ml_queue.get()

        try:

            (
                waist_ax,
                waist_ay,
                waist_az,
                thigh_ax,
                thigh_ay,
                thigh_az,
            ) = sample

            # ==================================================
            # RUN ML
            # ==================================================

            result = fall_ml.push(
                waist_ax,
                waist_ay,
                waist_az,
                thigh_ax,
                thigh_ay,
                thigh_az,
            )

            # --------------------------------------------------
            # No complete window yet.
            # --------------------------------------------------

            if result is None:

                continue

            # ==================================================
            # ML IS WORKING
            # ==================================================

            with ml_state_lock:

                ml_available = True

                # Store continuous probability.

                ml_probability = (
                    result.fall_decision
                )

                # ------------------------------------------------
                # USER REQUEST:
                #
                # FALL if probability > 0.52
                # ------------------------------------------------

                ml_fall_decision = (
                    result.fall_decision
                    > ML_FALL_THRESHOLD
                )

            # ==================================================
            # LOG RESULT
            # ==================================================

            log_stage3d6_result(
                result
            )

            # ==================================================
            # PRINT RESULT
            # ==================================================

            print("--------------------------------")
            print("STAGE 3D.6 ML RESULT")
            print(
                "Window index       :",
                result.window_index
            )
            print(
                f"W2 logit           : "
                f"{result.w2_logit:.6f}"
            )
            print(
                f"W2 probability     : "
                f"{result.w2_probability:.6f}"
            )
            print(
                f"T1 raw logit       : "
                f"{result.t1_raw_logit:.6f}"
            )
            print(
                f"T1 raw probability : "
                f"{result.t1_raw_probability:.6f}"
            )
            print(
                "T1 calibrated      : "
                f"{result.t1_calibrated_probability:.6f}"
            )
            print(
                "Fusion probability : "
                f"{result.fusion_probability:.6f}"
            )
            print(
                "EMA probability    : "
                f"{result.ema_probability:.6f}"
            )
            print(
                "ML threshold       : "
                f"{ML_FALL_THRESHOLD:.2f}"
            )
            print(
                "ML fall decision   :",
                ml_fall_decision
            ) 
            print(
                f"Inference latency  : "
                f"{result.inference_latency_ms:.3f} ms"
            )
            print("--------------------------------")

            # ==================================================
            # ML FALL DETECTED
            # ==================================================
            #
            # Only send once while probability remains above
            # threshold.
            #
            # Once probability falls below threshold, latch
            # resets and another fall can be detected.
            # ==================================================

            if ml_fall_decision:

                if not ml_fall_latched:

                    print("--------------------------------")
                    print(
                        "ML FALL DETECTED"
                    )
                    print(
                        f"EMA probability "
                        f"{result.ema_probability:.6f}"
                    )
                    print(
                        f"> threshold "
                        f"{ML_FALL_THRESHOLD:.2f}"
                    )
                    print("--------------------------------")

                    sent = send_to_esp32(
                        "FALL DETECTED"
                    )

                    if sent:

                        print("--------------------------------")
                        print(
                            "ML FALL DETECTED "
                            "sent to ESP32"
                        )
                        print("--------------------------------")

                    else:

                        print("--------------------------------")
                        print(
                            "ESP32 unavailable - "
                            "ML FALL DETECTED "
                            "could not be sent."
                        )
                        print("--------------------------------")

                    ml_fall_latched = True

            else:

                # Probability is back below threshold.
                # Permit a future fall event.

                ml_fall_latched = False

        except Exception as e:

            # ==================================================
            # ML FAILURE
            # ==================================================
            #
            # FSM is now permitted to operate.
            # ==================================================

            with ml_state_lock:

                ml_available = False

            print("--------------------------------")
            print("STAGE 3D.6 ML ERROR")
            print(
                "Error:",
                e
            )
            print(
                "ML marked unavailable."
            )
            print(
                "FSM FALLBACK ENABLED."
            )
            print("--------------------------------")

        finally:

            ml_queue.task_done()


# ==========================================================
# PROCESS ESP32 LINE
# ==========================================================

def process_esp32_line(
    line
):

    if not line:

        return

    # ======================================================
    # FALL TRUE
    # ======================================================

    if line == "FALL TRUE":

        process_fall_true()

        return

    # ======================================================
    # FALL FALSE
    # ======================================================

    if line == "FALL FALSE":

        process_fall_false()

        return

    # ======================================================
    # GPS
    # ======================================================

    if line.startswith("GPS"):

        process_gps_message(
            line
        )

        return

    # ======================================================
    # IMU
    # ======================================================

    process_imu_message(
        line
    )


# ==========================================================
# ESP32 RECEIVER THREAD
# ==========================================================

def esp32_receiver_thread():

    global sock
    global esp32_buffer

    timeout_count = 0

    MAX_TIMEOUTS = 3

    while esp32_thread_running:

        # ==================================================
        # GET CURRENT SOCKET
        # ==================================================

        with sock_lock:

            current_sock = sock

        # ==================================================
        # CONNECT IF NECESSARY
        # ==================================================

        if current_sock is None:

            current_sock = (
                connect_to_esp32()
            )

            if current_sock is None:

                time.sleep(1)

                continue

            esp32_buffer = ""

            timeout_count = 0

        # ==================================================
        # RECEIVE
        # ==================================================

        try:

            data = current_sock.recv(
                1024
            )

            timeout_count = 0

            # ==================================================
            # CONNECTION CLOSED
            # ==================================================

            if not data:

                print("--------------------------------")
                print("ESP32 CONNECTION LOST")
                print("Reconnecting...")
                print("--------------------------------")

                close_esp32_socket()

                esp32_buffer = ""

                timeout_count = 0

                continue

            esp32_buffer += (
                data.decode(
                    errors="ignore"
                )
            )

        # ==================================================
        # TIMEOUT
        # ==================================================

        except socket.timeout:

            timeout_count += 1

            print(
                "ESP32 timeout:",
                timeout_count,
                "/",
                MAX_TIMEOUTS
            )

            if (
                timeout_count
                >= MAX_TIMEOUTS
            ):

                print("--------------------------------")
                print("ESP32 CONNECTION LOST")
                print(
                    "No data received "
                    "for 3 seconds"
                )
                print("Reconnecting...")
                print("--------------------------------")

                close_esp32_socket()

                esp32_buffer = ""

                timeout_count = 0

            continue

        # ==================================================
        # SOCKET ERROR
        # ==================================================

        except (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            OSError
        ) as e:

            print("--------------------------------")
            print("ESP32 SOCKET ERROR")
            print(
                "Error:",
                e
            )
            print(
                "Reconnecting..."
            )
            print("--------------------------------")

            close_esp32_socket()

            esp32_buffer = ""

            timeout_count = 0

            continue

        # ==================================================
        # PROCESS COMPLETE LINES
        # ==================================================

        while "\n" in esp32_buffer:

            line, esp32_buffer = (
                esp32_buffer.split(
                    "\n",
                    1
                )
            )

            line = line.strip()

            if not line:

                continue

            try:

                process_esp32_line(
                    line
                )

            except Exception as e:

                print("--------------------------------")
                print(
                    "ESP32 DATA "
                    "PROCESSING ERROR"
                )
                print(
                    "Line:",
                    line
                )
                print(
                    "Error:",
                    e
                )
                print("--------------------------------")


# ==========================================================
# START WEATHER THREAD
# ==========================================================

threading.Thread(
    target=weather_thread,
    daemon=True
).start()


# ==========================================================
# START BLYNK
# ==========================================================

start_blynk()


# ==========================================================
# START SAM VOICE ASSISTANT
# ==========================================================

voice_thread = threading.Thread(
    target=start_voice_assistant,
    daemon=True
)

voice_thread.start()

print("--------------------------------")
print("Voice Assistant Started")
print("--------------------------------")


# ==========================================================
# START ML WORKER
# ==========================================================

ml_thread = threading.Thread(
    target=ml_worker_thread,
    daemon=True
)

ml_thread.start()

print("--------------------------------")
print("Stage 3D.6 ML Worker Started")
print("--------------------------------")

print("--------------------------------")
print(
    f"ML FALL THRESHOLD: "
    f"{ML_FALL_THRESHOLD}"
)
print(
    "ML is PRIMARY fall detector."
)
print(
    "FSM is FALLBACK only."
)
print("--------------------------------")


# ==========================================================
# START ESP32 COMMUNICATION
# ==========================================================

esp32_thread = threading.Thread(
    target=esp32_receiver_thread,
    daemon=True
)

esp32_thread.start()

print("--------------------------------")
print("ESP32 Communication Thread Started")
print("--------------------------------")


# ==========================================================
# RUN ARDUINO APP
# ==========================================================

App.run()
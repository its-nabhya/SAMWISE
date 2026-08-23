import time
import threading

import BlynkLib

from arduino.app_utils import Bridge


# ==========================================================
# BLYNK CONFIGURATION
# ==========================================================

BLYNK_TEMPLATE_ID = "INSERT YOUR TEMPLATE ID HERE"  # Replace with your Blynk template ID
BLYNK_TEMPLATE_NAME = "samwise"

BLYNK_AUTH_TOKEN = "Enter your Blynk auth token here"  # Replace with your Blynk auth token

BLYNK_SERVER = "blynk.cloud"
BLYNK_PORT = 80


# ==========================================================
# BLYNK VIRTUAL PINS
# ==========================================================

# V0 -> LED1 Switch
# V1 -> LED2 Switch
# V2 -> LED3 Switch
#
# V3 -> Weather Category
# V4 -> Weather Description

LED1_VPIN = 0
LED2_VPIN = 1
LED3_VPIN = 2

WEATHER_CATEGORY_VPIN = 3
WEATHER_DESCRIPTION_VPIN = 4


# ==========================================================
# PHYSICAL UNO Q PINS
# ==========================================================

LED1_PIN = "D4"
LED2_PIN = "D5"
LED3_PIN = "D6"


# ==========================================================
# DEVICE STATES
# ==========================================================

led1_state = 0
led2_state = 0
led3_state = 0


# ==========================================================
# WEATHER STATES
# ==========================================================

weather_category = "UNKNOWN"
weather_description = "UNKNOWN"


# ==========================================================
# THREAD LOCK
# ==========================================================

state_lock = threading.Lock()


# ==========================================================
# BLYNK OBJECT
# ==========================================================

blynk = None


# ==========================================================
# WEATHER UPDATE FUNCTION
# ==========================================================
#
# This function is called by main.py whenever new weather
# data is obtained from WeatherForecast.
#
# It does NOT directly call Blynk.
#
# Instead, it updates the shared values.
#
# The Blynk thread will send the values safely.
# ==========================================================

def update_weather_dashboard(category, description):

    global weather_category
    global weather_description

    with state_lock:

        weather_category = str(category)
        weather_description = str(description)

    print("--------------------------------")
    print("BLYNK WEATHER UPDATED")
    print("Category    :", weather_category)
    print("Description :", weather_description)
    print("--------------------------------")


# ==========================================================
# LED1 CONTROL
# ==========================================================

def set_led1(state):

    global led1_state

    state = 1 if int(state) else 0

    with state_lock:

        led1_state = state

    print("--------------------------------")
    print("LED1")
    print("Blynk V0 :", state)
    print("GPIO     :", LED1_PIN)
    print("State    :", "ON" if state else "OFF")
    print("--------------------------------")

    try:

        Bridge.call(
            "set_led1",
            state
        )

    except Exception as e:

        print("LED1 Bridge error:", e)


# ==========================================================
# LED2 CONTROL
# ==========================================================

def set_led2(state):

    global led2_state

    state = 1 if int(state) else 0

    with state_lock:

        led2_state = state

    print("--------------------------------")
    print("LED2")
    print("Blynk V1 :", state)
    print("GPIO     :", LED2_PIN)
    print("State    :", "ON" if state else "OFF")
    print("--------------------------------")

    try:

        Bridge.call(
            "set_led2",
            state
        )

    except Exception as e:

        print("LED2 Bridge error:", e)


# ==========================================================
# LED3 CONTROL
# ==========================================================

def set_led3(state):

    global led3_state

    state = 1 if int(state) else 0

    with state_lock:

        led3_state = state

    print("--------------------------------")
    print("LED3")
    print("Blynk V2 :", state)
    print("GPIO     :", LED3_PIN)
    print("State    :", "ON" if state else "OFF")
    print("--------------------------------")

    try:

        Bridge.call(
            "set_led3",
            state
        )

    except Exception as e:

        print("LED3 Bridge error:", e)


# ==========================================================
# BLYNK V0 CALLBACK -> LED1
# ==========================================================

def led1_handler(value):

    try:

        state = int(value[0])

        print("--------------------------------")
        print("BLYNK V0 RECEIVED")
        print("LED1 =", state)
        print("--------------------------------")

        set_led1(state)

    except Exception as e:

        print("LED1 Blynk error:", e)


# ==========================================================
# BLYNK V1 CALLBACK -> LED2
# ==========================================================

def led2_handler(value):

    try:

        state = int(value[0])

        print("--------------------------------")
        print("BLYNK V1 RECEIVED")
        print("LED2 =", state)
        print("--------------------------------")

        set_led2(state)

    except Exception as e:

        print("LED2 Blynk error:", e)


# ==========================================================
# BLYNK V2 CALLBACK -> LED3
# ==========================================================

def led3_handler(value):

    try:

        state = int(value[0])

        print("--------------------------------")
        print("BLYNK V2 RECEIVED")
        print("LED3 =", state)
        print("--------------------------------")

        set_led3(state)

    except Exception as e:

        print("LED3 Blynk error:", e)


# ==========================================================
# SEND CURRENT WEATHER TO BLYNK
# ==========================================================

def send_weather_to_blynk():

    if blynk is None:
        return

    with state_lock:

        category = weather_category
        description = weather_description

    try:

        blynk.virtual_write(
            WEATHER_CATEGORY_VPIN,
            category
        )

        blynk.virtual_write(
            WEATHER_DESCRIPTION_VPIN,
            description
        )

    except Exception as e:

        print("Weather Blynk update error:", e)


# ==========================================================
# BLYNK CONNECTION THREAD
# ==========================================================

def blynk_thread():

    global blynk

    print("--------------------------------")
    print("BLYNK THREAD STARTED")
    print("--------------------------------")

    while True:

        try:

            print("--------------------------------")
            print("CONNECTING TO BLYNK")
            print("Server:", BLYNK_SERVER)
            print("--------------------------------")

            # ------------------------------------------------
            # CREATE BLYNK OBJECT
            # ------------------------------------------------

            blynk = BlynkLib.Blynk(

                BLYNK_AUTH_TOKEN,

                server=BLYNK_SERVER,

                port=BLYNK_PORT

            )


            # ------------------------------------------------
            # REGISTER LED CALLBACKS
            # ------------------------------------------------

            blynk.on(
                "V0",
                led1_handler
            )

            blynk.on(
                "V1",
                led2_handler
            )

            blynk.on(
                "V2",
                led3_handler
            )


            print("--------------------------------")
            print("BLYNK CONNECTED")
            print("--------------------------------")


            # ------------------------------------------------
            # SEND INITIAL LED STATES
            # ------------------------------------------------

            with state_lock:

                current_led1 = led1_state
                current_led2 = led2_state
                current_led3 = led3_state

            blynk.virtual_write(
                LED1_VPIN,
                current_led1
            )

            blynk.virtual_write(
                LED2_VPIN,
                current_led2
            )

            blynk.virtual_write(
                LED3_VPIN,
                current_led3
            )


            # ------------------------------------------------
            # SEND CURRENT WEATHER
            # ------------------------------------------------

            send_weather_to_blynk()


            # ------------------------------------------------
            # RUN BLYNK
            # ------------------------------------------------

            last_weather_sent = 0

            while True:

                blynk.run()


                # --------------------------------------------
                # Send weather periodically
                #
                # This also makes sure that the latest weather
                # reaches Blynk after a reconnect.
                # --------------------------------------------

                if time.time() - last_weather_sent >= 2:

                    send_weather_to_blynk()

                    last_weather_sent = time.time()


                time.sleep(0.01)


        except Exception as e:

            print("--------------------------------")
            print("BLYNK CONNECTION ERROR")
            print("Error:", e)
            print("RETRYING IN 5 SECONDS")
            print("--------------------------------")

            blynk = None

            time.sleep(5)


# ==========================================================
# START BLYNK
# ==========================================================

def start_blynk():

    blynk_background_thread = threading.Thread(

        target=blynk_thread,

        daemon=True

    )

    blynk_background_thread.start()

    print("--------------------------------")
    print("BLYNK BACKGROUND THREAD STARTED")
    print("--------------------------------")
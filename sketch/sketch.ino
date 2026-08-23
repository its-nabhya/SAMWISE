#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include "weather_frames.h"


// ==========================================================
// PHYSICAL LED PINS
// ==========================================================

const int LED1_PIN = D4;
const int LED2_PIN = D5;
const int LED3_PIN = D6;


// ==========================================================
// LED MATRIX
// ==========================================================

Arduino_LED_Matrix matrix;


// ==========================================================
// LED1 CONTROL
// ==========================================================

void set_led1(int state)
{
    if (state)
    {
        digitalWrite(LED1_PIN, HIGH);
    }
    else
    {
        digitalWrite(LED1_PIN, LOW);
    }
}


// ==========================================================
// LED2 CONTROL
// ==========================================================

void set_led2(int state)
{
    if (state)
    {
        digitalWrite(LED2_PIN, HIGH);
    }
    else
    {
        digitalWrite(LED2_PIN, LOW);
    }
}


// ==========================================================
// LED3 CONTROL
// ==========================================================

void set_led3(int state)
{
    if (state)
    {
        digitalWrite(LED3_PIN, HIGH);
    }
    else
    {
        digitalWrite(LED3_PIN, LOW);
    }
}


// ==========================================================
// SETUP
// ==========================================================

void setup()
{
    // ------------------------------------------------------
    // LED PINS
    // ------------------------------------------------------

    pinMode(LED1_PIN, OUTPUT);
    pinMode(LED2_PIN, OUTPUT);
    pinMode(LED3_PIN, OUTPUT);


    // ------------------------------------------------------
    // TURN ALL LEDs OFF INITIALLY
    // ------------------------------------------------------

    digitalWrite(LED1_PIN, LOW);
    digitalWrite(LED2_PIN, LOW);
    digitalWrite(LED3_PIN, LOW);


    // ------------------------------------------------------
    // LED MATRIX
    // ------------------------------------------------------

    matrix.begin();
    matrix.clear();


    // ------------------------------------------------------
    // ARDUINO BRIDGE
    // ------------------------------------------------------

    Bridge.begin();


    // ------------------------------------------------------
    // FUNCTIONS CALLABLE FROM PYTHON
    // ------------------------------------------------------

    Bridge.provide(
        "set_led1",
        set_led1
    );

    Bridge.provide(
        "set_led2",
        set_led2
    );

    Bridge.provide(
        "set_led3",
        set_led3
    );
}


// ==========================================================
// LOOP
// ==========================================================

void loop()
{
    String weather;

    bool ok =
        Bridge.call("get_weather_forecast")
              .result(weather);


    if (ok)
    {
        if (weather == "sunny")
        {
            matrix.loadSequence(sunny);
            playRepeat(10);
        }

        else if (weather == "cloudy")
        {
            matrix.loadSequence(cloudy);
            playRepeat(10);
        }

        else if (weather == "rainy")
        {
            matrix.loadSequence(rainy);
            playRepeat(20);
        }

        else if (weather == "snowy")
        {
            matrix.loadSequence(snowy);
            playRepeat(10);
        }

        else if (weather == "foggy")
        {
            matrix.loadSequence(foggy);
            playRepeat(5);
        }
    }

    delay(1000);
}


// ==========================================================
// WEATHER ANIMATION
// ==========================================================

void playRepeat(int repeat_count)
{
    for (int i = 0; i < repeat_count; i++)
    {
        matrix.playSequence();
    }
}
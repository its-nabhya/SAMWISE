#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <TinyGPS++.h>

//=====================================================
// WiFi
//=====================================================

const char* ssid = "Galaxy A35 5G 9940";
const char* password = "vevy3tmd";

WiFiServer server(5000);
WiFiClient client;

//=====================================================
// PCA9548A
//=====================================================

#define PCA9548A_ADDR 0x70

Adafruit_MPU6050 waistMPU;
Adafruit_MPU6050 thighMPU;

//=====================================================
// GPS
//=====================================================

TinyGPSPlus gps;
HardwareSerial GPS(2);

#define GPS_RX 16
#define GPS_TX 17

const double DEFAULT_LAT = 22.494510;
const double DEFAULT_LON = 88.325614;

//=====================================================
// Alarm Hardware
//=====================================================

#define BUZZER_PIN 18
#define BUTTON_PIN 19

//=====================================================
// Timers
//=====================================================

unsigned long lastSend = 0;
unsigned long lastGPSSend = 0;

//=====================================================
// Fall Detection State Machine
//=====================================================

enum FallState
{
    NORMAL,
    FREE_FALL,
    IMPACT
};

FallState state = NORMAL;

unsigned long freeFallTime = 0;
unsigned long impactTime = 0;

const float FREE_FALL_THRESHOLD = 4.0;
const float IMPACT_THRESHOLD    = 10.0;
const float STILL_THRESHOLD     = 1.5;

const unsigned long IMPACT_WINDOW = 1000;
const unsigned long STILL_TIME    = 1500;

//=====================================================
// Alarm State Machine
//=====================================================

enum AlarmState
{
    ALARM_IDLE,
    ALARM_PREALARM,
    ALARM_ACTIVE
};

AlarmState alarmState = ALARM_IDLE;

unsigned long preAlarmStart = 0;
unsigned long alarmStart = 0;

bool firstBeepDone = false;

//=====================================================
// Buzzer Control
//=====================================================

bool buzzerState = false;
unsigned long lastBuzzerToggle = 0;

const unsigned long FIRST_BEEP_TIME = 1000;
const unsigned long CANCEL_WINDOW   = 8000;
const unsigned long MAX_ALARM_TIME  = 300000;
const unsigned long BUZZ_INTERVAL   = 500;

//=====================================================
// Button Debounce
//=====================================================

bool lastButtonState = LOW;
bool buttonPressed = false;

unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

//=====================================================
// PCA9548A Channel Selection
//=====================================================

void selectChannel(uint8_t channel)
{
    Wire.beginTransmission(PCA9548A_ADDR);
    Wire.write(1 << channel);
    Wire.endTransmission();
}

//=====================================================
// Read Button (Debounced)
//=====================================================

bool readButton()
{
    bool reading = digitalRead(BUTTON_PIN);

    if(reading != lastButtonState)
    {
        lastDebounceTime = millis();
    }

    if((millis() - lastDebounceTime) > debounceDelay)
    {
        if(reading == HIGH && !buttonPressed)
        {
            buttonPressed = true;
            lastButtonState = reading;
            return true;
        }

        if(reading == LOW)
        {
            buttonPressed = false;
        }
    }

    lastButtonState = reading;

    return false;
}

//=====================================================
// Continuous Buzzer
//=====================================================

void updateBuzzer()
{
    if(millis() - lastBuzzerToggle >= BUZZ_INTERVAL)
    {
        lastBuzzerToggle = millis();

        buzzerState = !buzzerState;

        digitalWrite(BUZZER_PIN, buzzerState);
    }
}

//=====================================================
// Stop Alarm
//=====================================================

void stopAlarm()
{
    digitalWrite(BUZZER_PIN, LOW);

    buzzerState = false;

    alarmState = ALARM_IDLE;

    firstBeepDone = false;
}

//=====================================================
// Setup
//=====================================================

void setup()
{
    delay(1000);

    Serial.begin(115200);

    delay(1000);

    Wire.begin(21,22);

    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    pinMode(BUTTON_PIN, INPUT);

    GPS.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);

    selectChannel(0);

    if(!waistMPU.begin())
    {
        Serial.println("Waist MPU not found!");
    }

    selectChannel(1);

    if(!thighMPU.begin())
    {
        Serial.println("Thigh MPU not found!");
    }

    WiFi.begin(ssid, password);

    while(WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println();

    Serial.print("IP: ");

    Serial.println(WiFi.localIP());

    server.begin();

    Serial.println("ESP32 Ready");
}
//=====================================================
// LOOP
//=====================================================

void loop()
{
    // -----------------------------------------------
    // Read GPS continuously
    // -----------------------------------------------
    while (GPS.available())
    {
        gps.encode(GPS.read());
    }

    // -----------------------------------------------
    // Accept WiFi client
    // -----------------------------------------------
    if (!client || !client.connected())
    {
        client = server.available();
        return;
    }

    // -----------------------------------------------
    // Read IMUs every 10 ms
    // -----------------------------------------------
    if (millis() - lastSend >= 10)
    {
        lastSend = millis();

        sensors_event_t wa, wg, wt;
        sensors_event_t ta, tg, tt;

        // Waist MPU
        selectChannel(0);
        waistMPU.getEvent(&wa, &wg, &wt);

        // Thigh MPU
        selectChannel(1);
        thighMPU.getEvent(&ta, &tg, &tt);

        // -------------------------------------------
        // Send IMU CSV
        // -------------------------------------------

        String csv =
            String(wa.acceleration.x, 3) + "," +
            String(wa.acceleration.y, 3) + "," +
            String(wa.acceleration.z, 3) + "," +
            String(wg.gyro.x, 3) + "," +
            String(wg.gyro.y, 3) + "," +
            String(wg.gyro.z, 3) + "," +
            String(ta.acceleration.x, 3) + "," +
            String(ta.acceleration.y, 3) + "," +
            String(ta.acceleration.z, 3) + "," +
            String(tg.gyro.x, 3) + "," +
            String(tg.gyro.y, 3) + "," +
            String(tg.gyro.z, 3);

        client.println(csv);
        Serial.println(csv);

        // -------------------------------------------
        // Magnitude of acceleration
        // -------------------------------------------

        float waistAcc =
            sqrt(
                wa.acceleration.x * wa.acceleration.x +
                wa.acceleration.y * wa.acceleration.y +
                wa.acceleration.z * wa.acceleration.z);

        float thighAcc =
            sqrt(
                ta.acceleration.x * ta.acceleration.x +
                ta.acceleration.y * ta.acceleration.y +
                ta.acceleration.z * ta.acceleration.z);

        float waistMove = fabs(waistAcc - 9.81);
        float thighMove = fabs(thighAcc - 9.81);

        // -------------------------------------------
        // Fall Detection State Machine
        // -------------------------------------------

        switch (state)
        {
            case NORMAL:

                if (waistAcc < FREE_FALL_THRESHOLD &&
                    thighAcc < FREE_FALL_THRESHOLD)
                {
                    state = FREE_FALL;
                    freeFallTime = millis();

                    Serial.println("FREE FALL");
                }

                break;

            case FREE_FALL:

                if (millis() - freeFallTime > IMPACT_WINDOW)
                {
                    state = NORMAL;
                }

                else if (waistAcc > IMPACT_THRESHOLD &&
                         thighAcc > IMPACT_THRESHOLD)
                {
                    state = IMPACT;
                    impactTime = millis();

                    Serial.println("IMPACT");
                }

                break;

            case IMPACT:

                if ((millis() - impactTime > STILL_TIME) &&
                    (waistMove < STILL_THRESHOLD) &&
                    (thighMove < STILL_THRESHOLD))
                {
                    Serial.println("FALL DETECTED");

                    // ----------------------------------
                    // Start Pre-Alarm
                    // ----------------------------------

                    alarmState = ALARM_PREALARM;

                    preAlarmStart = millis();

                    firstBeepDone = false;

                    state = NORMAL;
                }

                if (millis() - impactTime > 5000)
                {
                    state = NORMAL;
                }

                break;
        }
    }

    //=================================================
    // Alarm State Machine
    //=================================================

    switch(alarmState)
    {

    //-------------------------------------------------
    // Idle
    //-------------------------------------------------

    case ALARM_IDLE:
        break;

    //-------------------------------------------------
    // 8 second confirmation period
    //-------------------------------------------------

    case ALARM_PREALARM:

        // -------- First 1 second beep --------

        if(!firstBeepDone)
        {
            if(millis() - preAlarmStart <= FIRST_BEEP_TIME)
            {
                digitalWrite(BUZZER_PIN, HIGH);
            }
            else
            {
                digitalWrite(BUZZER_PIN, LOW);
                firstBeepDone = true;
            }
        }

        // -------- User cancelled --------

        if(readButton())
        {
            Serial.println("False Alarm");

            if(client.connected())
            {
                client.println("FALL FALSE");
            }

            stopAlarm();
            break;
        }

        // -------- 8 seconds elapsed --------

        if(millis() - preAlarmStart >= CANCEL_WINDOW)
        {
            Serial.println("Fall Confirmed");

            if(client.connected())
            {
                client.println("FALL TRUE");
            }

            alarmState = ALARM_ACTIVE;

            alarmStart = millis();

            buzzerState = false;
            lastBuzzerToggle = millis();
        }

        break;

    //-------------------------------------------------
    // Confirmed Alarm
    //-------------------------------------------------

    case ALARM_ACTIVE:

        updateBuzzer();

        // User acknowledged

        if(readButton())
        {
            Serial.println("Alarm Acknowledged");

            stopAlarm();

            break;
        }

        // 5 minute timeout

        if(millis() - alarmStart >= MAX_ALARM_TIME)
        {
            Serial.println("Alarm Timeout");

            stopAlarm();
        }

        break;
    }

    //=================================================
    // GPS Transmission
    //=================================================

    if(millis() - lastGPSSend >= 1000)
    {
        lastGPSSend = millis();

        double lat = DEFAULT_LAT;
        double lon = DEFAULT_LON;

        if(gps.location.isValid())
        {
            lat = gps.location.lat();
            lon = gps.location.lng();
        }

        String gpsMsg =
            "GPS," +
            String(lat,6) + "," +
            String(lon,6);

        if(client.connected())
        {
            client.println(gpsMsg);
        }

        Serial.println(gpsMsg);
    }

    //=================================================
    // Receive Messages from UNO Q
    //=================================================

    while(client.available())
    {
        String msg = client.readStringUntil('\n');

        msg.trim();

        Serial.print("UNO Q says: ");
        Serial.println(msg);

        // Future commands can be added here

        /*
        if(msg=="RESET")
        {
            stopAlarm();
        }

        if(msg=="BUZZ")
        {
            alarmState=ALARM_ACTIVE;
        }
        */
    }

}
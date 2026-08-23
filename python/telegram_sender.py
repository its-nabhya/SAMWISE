"""
telegram_sender.py

Telegram notification module for Fall Detection System
"""

import requests

# ============================================================
# Telegram Configuration
# ============================================================

BOT_TOKEN = "INSERT YOUR BOT TOKEN HERE"  # Replace with your bot token

# Add every person who should receive notifications
CHAT_IDS = [
    #Enter your chat IDs here, for example:
    #174165227, 
    #987654321,  
]

# ============================================================
# Telegram Notification Function
# ============================================================

def send_telegram(fall_time, latitude, longitude):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    message = f"""🚨 EMERGENCY FALL ALERT 🚨

A fall has been detected.

🕒 Time:
{fall_time}

📍 Latitude:
{latitude}

📍 Longitude:
{longitude}

🗺 Google Maps:
https://maps.google.com/?q={latitude},{longitude}

Please check on the person immediately.

Arduino Uno Q Fall Detection System
"""

    success = True

    for chat_id in CHAT_IDS:

        try:

            response = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": message
                },
                timeout=10
            )

            if response.status_code == 200:

                print(f"Telegram sent to {chat_id}")

            else:

                print(f"Telegram failed ({chat_id})")
                print(response.text)

                success = False

        except Exception as e:

            print(f"Telegram Error ({chat_id})")
            print(e)

            success = False

    return success
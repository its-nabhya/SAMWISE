import smtplib
from email.mime.text import MIMEText


SENDER_EMAIL = "samwisefalldetectionsystem@gmail.com" # Replace with your email address
APP_PASSWORD = "EMAIL_APP_PASSWORD" # Replace with your app password
RECIPIENTS = [
    "samwisefalldetectionsystem@gmail.com"
]

def send_email(fall_time, latitude, longitude):

    subject = "Emergency Fall Detection Alert"

    body = f"""Emergency Fall Alert

A fall has been detected.

Time:
{fall_time}

Latitude:
{latitude}

Longitude:
{longitude}

Google Maps:
https://maps.google.com/?q={latitude},{longitude}
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECIPIENTS)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(
        SENDER_EMAIL,
        RECIPIENTS,
        msg.as_string()
    )
    server.quit()

    return True
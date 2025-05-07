

import grovepi
# Import smtplib for the actual sending function
import smtplib, string, subprocess, time

import requests


print("System Working")
switch = 4
led_status = 3
buzzer = 5
ultrasonic_port = 2

# Change for local IP address
REMOTE_URL   = "http://192.168.98.15:5001/trigger"
HEADERS = {}

grovepi.pinMode(led_status,"OUTPUT")
grovepi.pinMode(buzzer,    "OUTPUT")
grovepi.pinMode(switch,"INPUT")

while True:     # in case of IO error, restart
    try:
        while True:
            if grovepi.digitalRead(switch) == 1:    # If the system is ON
                if grovepi.ultrasonicRead(ultrasonic_port) < 50:  # If a person walks through the door
                    print("Welcome")
                    grovepi.analogWrite(buzzer,100) # Make a sound on the Buzzer
                    time.sleep(.5)
                    grovepi.analogWrite(buzzer,0)       # Turn off the Buzzer
                    resp = requests.post(REMOTE_URL, headers=HEADERS, timeout=5)

                    print("→ Remote responded:", resp.status_code, resp.json())


    except IOError:
        print("Error")
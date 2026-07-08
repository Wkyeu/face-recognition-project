import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import cv2 #handle image IO, image preprocessing(cropping,resizing), connect to webcam and capture videos
from datetime import datetime
from deepface import DeepFace

DB_PATH = "database"
LOG_FILE = "attendance_log.csv"
COOLDOWN_MINUTES = 10

#Create the CSV with headers if it does not exist yet
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Timestamp"])

last_seen ={}

def already_logged_recently(name):
    if name not in last_seen:
        return False
    elapsed = (datetime.now() - last_seen[name]).total_seconds() / 60
    return elapsed < COOLDOWN_MINUTES

def log_attendance(name):
    now = datetime.now()
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S")])
    last_seen[name] = now
    print(f"Logged: {name} at {now.strftime('%H:%M:%S')}")

cap = cv2.VideoCapture(0)
frame_count = 0

print("Starting attendance system..... press 'q' to quit")



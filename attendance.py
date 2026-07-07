import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import cv2
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
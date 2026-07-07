import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import cv2
from deepface import DeepFace

DB_PATH = "database"
LOG_FILE = "attendance_log.csv"

#Create the CSV with headers if it does not exist yet
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Timestamp"])
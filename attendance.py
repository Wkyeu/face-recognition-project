import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from deepface import DeepFace

DB_PATH = "database"
LOG_FILE = "attendance_log.csv"
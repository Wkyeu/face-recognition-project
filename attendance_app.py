#app using streamlit (reruns the file everytime interaction is detected)

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
from datetime import datetime 
import cv2
import numpy as np #to convert photo from streamlit into a openCV format
import pandas as pd #displa/sort csv as a proper table
import streamlit as st 
from deepface import DeepFace 

DB_PATH = "database"
LOG_FILE = "attendance_log.csv"
COOLDOWN_MINUTES = 10
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"

os.makedirs(DB_PATH, exist_ok=True) #creates database folder if it does not already exist

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Timestamp"])

#------------------Helper---------------------------------

def load_log_df():
    return pd.read_csv(LOG_FILE) #read the csv file and return it as a dataframe

def already_logged_recently(name, log_df):
    matches = log_df[log_df["Name"] == name] #log_df["Name"] == name:checls every row's Name column against the name we are looking for, returns True/False for each row. log_df[...]: using the True/False list to filter the table down to just the matching rows
    if matches.empty:
        return False
    last_time = pd.to_datetime(matches["Timestamp"]).max() #.max() grabs the most recent timestamp if there are multiple past entries of the same name
    elapsed = (datetime.now() - last_time).total_seconds() / 60
    return elapsed < COOLDOWN_MINUTES

def log_attendance(name):
    now = datetime.now()
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S")])

#used to delete any .pkl files (from deepface caching embeddings) so deepface have to rebuild it the next time
def rebuild_cache_if_needed():
    for f in os.listdir(DB_PATH):
        if f.endswith(".pkl"):
            os.remove(os.path.join(DB_PATH, f))

#-------------------Page Setup-----------------------------

#sets basic page metadata
st.set_page_config(page_title="Face Attendance", page_icon="🧑‍💻", layout="centered")

#sets a heading at the top of page (Note: different from page_title in set_page_config as that is for the tab title)
st.title("🧑‍💻 Face Recognition Attendance")

#st.tabs() creates clickable tabs on top of the page
tab_checkin, tab_database, tab_log = st.tabs(["Check In", "Manage Database", "Attendance Log"])




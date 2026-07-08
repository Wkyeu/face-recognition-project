import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import cv2 #handle image IO, image preprocessing(cropping,resizing), connect to webcam and capture videos
from datetime import datetime
from deepface import DeepFace

DB_PATH = "database"
LOG_FILE = "attendance_log.csv"
COOLDOWN_MINUTES = 10
CHECK_INTERVAL = 60

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

while True:
    ret, frame = cap.read() #return a tuple with 2 var, ret: Boolean to check if webcam is working properly, frame: 3D array containing raw pixel data
    if not ret:
        break 
    
    frame_count += 1
    cv2.imshow("Attendance Camera", frame) #imageshow --- "Attendance Camera": String, name given to the popup window on the top left title bar, frame: Image array, grid of pixels

    if frame_count % CHECK_INTERVAL == 0:
        try:
            results = DeepFace.find(
                img_path=frame,
                db_path=DB_PATH,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=False,
                silent=True #supress deepface logging messages
            )

            for df in results:
                if len(df) > 0:
                    best_match = df.iloc[0] #integer location: grab the first row of dataframe
                    if best_match["distance"] < 0.4: 
                        name = os.path.basename(best_match["identity"])
                        if not already_logged_recently(name):
                            log_attendance(name)
        
        except Exception as e:
            print(f"Recognition skipped this frame: {e}")
        
    if cv2.waitKey(1) & 0xFF == ord('q'): #.waitKey(1) tells python to pause for 1 millisecond and detect keystroke, 0xFF==ord('q') checks if q key is pressed to break
        break 
    
    



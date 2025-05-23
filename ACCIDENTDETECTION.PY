import torch
from ultralytics import YOLO
import cv2
from datetime import datetime, timedelta
from twilio.rest import Client
import pygame
import csv
import os

# === CONFIGURATION ===
COOLDOWN = timedelta(minutes=5)
CLEAR_PERIOD = timedelta(seconds=30)
DETECTION_DURATION = 5  # seconds

# Twilio credentials (REPLACE THESE WITH YOUR OWN SECURELY)
ACCOUNT_SID = "YOUR_TWILIO_ACCOUNT_SID"
AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"
TWILIO_PHONE = "YOUR_TWILIO_PHONE_NUMBER"
USER_PHONE = "RECEIVER_PHONE_NUMBER"
LOCATION_LINK = "https://maps.app.goo.gl/YOUR_LOCATION_LINK"

# File paths (Ensure these paths exist)
MODEL_PATH = r"path\to\your\best.pt"
ALARM_PATH = r"path\to\your\alarm.wav"
CSV_PATH = r"path\to\your\csv\accident_log.csv"
VIDEO_DIR = r"path\to\your\output_frames"

# === SETUP ===
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(ALARM_PATH)

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# === Monitor Class ===
class AccidentMonitor:
    def __init__(self):
        self.last_alert = None
        self.detection_start = None
        self.clear_start = None
        self.alert_sent = False
        self.recording = False
        self.video_writer = None

    def should_alert(self, now):
        return (
            not self.alert_sent and
            self.detection_duration(now) >= DETECTION_DURATION and
            (self.last_alert is None or (now - self.last_alert) >= COOLDOWN) and
            self.scene_cleared(now)
        )

    def detection_duration(self, now):
        return (now - self.detection_start).total_seconds() if self.detection_start else 0

    def scene_cleared(self, now):
        return (
            self.last_alert is None or
            (self.clear_start and (now - self.clear_start) >= CLEAR_PERIOD)
        )

monitor = AccidentMonitor()

# === Functions ===
def send_alert():
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            body=f"⚠️ Accident detected! Immediate attention required!\nLocation: {LOCATION_LINK}",
            from_=TWILIO_PHONE,
            to=USER_PHONE
        )
        print("[ALERT] SMS sent.")
        alarm_sound.play()
    except Exception as e:
        print("[ERROR] Twilio:", e)

def start_recording(now):
    if not monitor.recording:
        filename = os.path.join(VIDEO_DIR, f"accident_{now.strftime('%Y%m%d_%H%M%S')}.mp4")
        monitor.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        monitor.recording = True
        print("[INFO] Recording started:", filename)

def log_to_csv(now):
    header = ["Timestamp", "Message", "Location"]
    row = [now.strftime("%Y-%m-%d %H:%M:%S"), "Accident detected", LOCATION_LINK]
    new_file = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)
    print("[LOG] Event recorded in CSV.")

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = datetime.now()
    resized = cv2.resize(frame, (256, 256))
    results = model.predict(resized)
    detections = results[0].boxes

    accident_present = False
    for box in detections:
        if box.conf > 0.3:
            accident_present = True

            # Scale box to original size
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = int(x1 * (frame.shape[1] / 256))
            y1 = int(y1 * (frame.shape[0] / 256))
            x2 = int(x2 * (frame.shape[1] / 256))
            y2 = int(y2 * (frame.shape[0] / 256))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "ACCIDENT", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # === Detection Logic ===
    if accident_present:
        monitor.clear_start = None
        if monitor.detection_start is None:
            monitor.detection_start = current_time

        if monitor.should_alert(current_time):
            send_alert()
            log_to_csv(current_time)
            start_recording(current_time)
            monitor.last_alert = current_time
            monitor.alert_sent = True
    else:
        if monitor.clear_start is None:
            monitor.clear_start = current_time

        if monitor.last_alert and (
            (current_time - monitor.last_alert >= COOLDOWN) and
            (current_time - monitor.clear_start >= CLEAR_PERIOD)
        ):
            print("[RESET] Monitor reset.")
            monitor = AccidentMonitor()

    # === Save Frame to Video ===
    if monitor.recording and monitor.video_writer:
        monitor.video_writer.write(frame)

    # === Display Window ===
    cv2.imshow("Accident Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
if monitor.video_writer:
    monitor.video_writer.release()
cap.release()
cv2.destroyAllWindows()
pygame.quit()

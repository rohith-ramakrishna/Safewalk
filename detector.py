import time
import cv2
import pyttsx3
from ultralytics import YOLO
import numpy as np
print("Loading AI model...")
detector = YOLO("yolov8n.pt")

segmenter = YOLO(
    r"C:\Users\aksha\Downloads\runs\segment\train-2\weights\best.pt"
)
print("✅ AI Model loaded!")

engine = pyttsx3.init()
engine.setProperty('rate', 150
last_alert_time = 0
alert_cooldown = 3

HAZARD_LABELS = [
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "pothole",
    "bench",
    "chair",
]

def speak_alert(label):
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time > alert_cooldown:
        message = f"Warning! {label} detected ahead. Please be careful."
        print(f"🔊 Speaking: {message}")
        engine.say(message)
        engine.runAndWait()
        last_alert_time = current_time

print("Starting camera...")
print("Press Q to quit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    seg_results = segmenter(frame)

    footpath_mask = None

    if seg_results[0].masks is not None:
        mask = seg_results[0].masks.data[0].cpu().numpy()

        footpath_mask = (mask > 0.5).astype("uint8")
    if not ret:
        print("❌ Camera error")
        break

    results = detector(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence > 0.5:
                label = detector.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if label in HAZARD_LABELS:

                    if footpath_mask is not None:

                        if cy < footpath_mask.shape[0] and cx < footpath_mask.shape[1]:

                            if footpath_mask[cy, cx] == 1:
                                color = (0, 0, 255)

                                speak_alert(label)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {confidence:.0%}"
                cv2.putText(frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, "SafeWalk - Hazard Detector",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (255, 255, 255), 2)

    cv2.imshow("SafeWalk", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")
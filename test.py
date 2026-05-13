import cv2
import torch
from ultralytics import YOLO
import pyttsx3

print("✅ OpenCV version:", cv2.__version__)
print("✅ PyTorch version:", torch.__version__)
print("✅ All libraries working!")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera detected!")
    cap.release()
else:
    print("❌ No camera found")


engine = pyttsx3.init()
engine.say("SafeWalk is ready!")
engine.runAndWait()
print("✅ Voice working!")

print("")
print("🚀 All systems go! Ready to build SafeWalk!")
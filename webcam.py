from ultralytics import YOLO
import cv2
model=YOLO("industrialsafety.pt")
results=model(source=0,show=True,conf=0.3,save=True)
cv2.imshow(results)
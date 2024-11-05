import cv2
from ultralytics import YOLOv10
import numpy as np
import math
import re
import os
import sqlite3
import json
from datetime import datetime
from paddleocr import PaddleOCR
import streamlit as st
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize YOLOv10 Model
model = YOLOv10("weights/best.pt")

# Class Names
className = ["License"]

# Initialize Paddle OCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)


# Function for OCR detection
def paddle_ocr(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]
    result = ocr.ocr(frame, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]
    pattern = re.compile('[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")
    return str(text)


# Create a table in the database if it doesn't exist
def create_table():
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS LicensePlates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            license_plate TEXT
        )
        '''
    )
    conn.commit()
    conn.close()


# Save detected plates into the database
def save_to_database(license_plates, start_time, end_time):
    create_table()  # Ensure table exists
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    for plate in license_plates:
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
    conn.commit()
    conn.close()


# Streamlit application setup
st.title("License Plate Detection App")

# Option to select between video/image or camera
option = st.selectbox("Select Input Source", ("Upload a video or image", "Use Camera"))

if option == "Upload a video or image":
    # Upload a video or image
    uploaded_file = st.file_uploader("Upload a video or image", type=["mp4", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load the uploaded file
        if uploaded_file.type in ["image/png", "image/jpg", "image/jpeg"]:
            # Handle image file
            image = Image.open(uploaded_file)
            frame = np.array(image)
        else:
            # Handle video file
            tfile = open("temp_video.mp4", 'wb')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture("temp_video.mp4")

            license_plates = set()
            startTime = datetime.now()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                currentTime = datetime.now()

                # YOLOv10 model prediction
                results = model.predict(frame, conf=0.45)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        label = paddle_ocr(frame, x1, y1, x2, y2)
                        if label:
                            license_plates.add(label)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                        c2 = x1 + textSize[0], y1 - textSize[1] - 3
                        cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)

            if (currentTime - startTime).seconds >= 20:
                endTime = currentTime
                save_to_database(license_plates, startTime, endTime)
                startTime = currentTime
                license_plates.clear()

            st.image(frame, channels="BGR")
            cap.release()

elif option == "Use Camera":
    # Access the camera and perform real-time detection
    st.text("Turn on the camera and detect license plates in real-time")
    run_camera = st.button("Start Camera")

    if run_camera:
        cap = cv2.VideoCapture(0)  # Open default camera
        license_plates = set()
        startTime = datetime.now()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access the camera")
                break

            currentTime = datetime.now()

            # YOLOv10 model prediction
            results = model.predict(frame, conf=0.45)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    label = paddle_ocr(frame, x1, y1, x2, y2)
                    if label:
                        license_plates.add(label)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            if (currentTime - startTime).seconds >= 20:
                endTime = currentTime
                save_to_database(license_plates, startTime, endTime)
                startTime = currentTime
                license_plates.clear()

            st.image(frame, channels="BGR")
        cap.release()

# Option to view stored license plates
if st.button("Show Saved License Plates"):
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM LicensePlates")
    rows = cursor.fetchall()
    for row in rows:
        st.write(f"License Plate: {row[3]}, Start Time: {row[1]}, End Time: {row[2]}")
    conn.close()


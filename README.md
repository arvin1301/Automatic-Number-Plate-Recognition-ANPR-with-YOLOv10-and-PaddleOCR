 # Automatic-Number-Plate-Recognition-ANPR-with-YOLOv10-and-PaddleOCR



This project implements an Automatic Number Plate Recognition (ANPR) system using YOLOv10 for object detection and PaddleOCR for optical character recognition (OCR). It can process both video files and live camera feeds to detect and recognize license plates.




# Features
-Real-time License Plate Detection: Use a webcam to detect and recognize license plates in real-time.
-Video Processing: Upload video files to process and detect license plates.
-Data Storage: Detected license plates, along with timestamps, are saved to a SQLite database and JSON files.
-Visualization: Displays detection results, including bounding boxes and recognized license plates on images.





# Table of Contents
-Prerequisites
-Installation
-Usage
-Database Structure
-License




# Prerequisites
Before you begin, ensure you have met the following requirements:
-Python 3.10 or higher
-Git
-A working webcam (for real-time detection)
-OpenCV
-YOLOv10 and PaddleOCR packages




# Installation
Clone the repository:
-git clone https://github.com/yourusername/ANPR.git
-cd ANPR




# Install the required packages:
-pip install -r requirements.txt




# Download the YOLOv10 weights:
-The weights will be downloaded automatically upon running the code. Alternatively, you can manually download them from YOLOv10 Releases.




# Install the Roboflow package for dataset management:
-pip install roboflow




# Usage
Run the application:
-To start the application, use the following command in your terminal:
-streamlit run app.py




Select Input Source:
-Choose between "Upload a video or image" or "Use Camera" from the dropdown menu.




Upload a Video/Image:
-If you select "Upload a video or image," you can upload a video file (MP4) or an image (PNG, JPG, JPEG) for processing.
-The application will display the processed frame with detected license plates.




Use Camera
-If you select "Use Camera," click the "Start Camera" button to begin real-time detection using your webcam.




View Saved License Plates:
-You can view all detected license plates stored in the SQLite database by clicking the "Show Saved License Plates" button.




Database Structure:
-The application creates a SQLite database named licensePlatesDatabase.db with the following table:




# LicensePlates Table:
Column Name	        Data Type	      Description
id	                 INTEGER        	Primary key (auto-increment)
start_time	         TEXT	           Timestamp when detection starts
end_time	           TEXT	           Timestamp when detection ends
license_plate  	    TEXT	           Detected license plate number





# Acknowledgements:
-YOLOv10 for the object detection model.
-PaddleOCR for optical character recognition.
-OpenCV for computer vision tasks.
-Streamlit for building the web app interface.



# Screenshot
![ANPR2](https://github.com/user-attachments/assets/37ef4631-1992-4ae1-9a8a-c81ec97dc293)
![ANPR1](https://github.com/user-attachments/assets/4818e93d-bda4-4b25-9d9f-d0b71cde8be2)


Overview:
This project implements a real-time object detection and alert system using computer vision and deep learning techniques. It includes three main functionalities:

Accident Detection: Detects accidents in video feeds and sends alerts to a control room.

Flood Detection: Detects flood events in video feeds and sends alerts.

Wild Animal Detection: Detects the presence of wild animals in video feeds and sends alerts.

The project utilizes YOLO (You Only Look Once) models for object detection and Twilio for sending SMS alerts.

---------------------------------------------------------------------------------------------------------------
Prerequisites:
Ensure you have the following installed:
Python 3.6+
OpenCV
NumPy
Twilio
Ultralyitcs YOLO

---------------------------------------------------------------------------------------------------------
Download YOLO Models

Place your YOLO models (accident7epochs.pt, animal.pt, Flood.pt) in the specified directory (/content/drive/MyDrive/models/).

--------------------------------------------------------------------------------------------------------
Set Up Twilio Credentials

Replace the placeholders in the script with your actual Twilio account SID, Auth Token, Twilio number, and recipient number.

# Twilio credentials for thread 1 (Accident detection)
thread_1_account_sid = 'YOUR_ACCOUNT_SID'
thread_1_auth_token = 'YOUR_AUTH_TOKEN'
thread_1_twilio_number = 'YOUR_TWILIO_NUMBER'
thread_1_recipient_number = 'RECIPIENT_NUMBER'

# Twilio credentials for thread 2 (Animal detection)
thread_2_account_sid = 'YOUR_ACCOUNT_SID'
thread_2_auth_token = 'YOUR_AUTH_TOKEN'
thread_2_twilio_number = 'YOUR_TWILIO_NUMBER'
thread_2_recipient_number = 'RECIPIENT_NUMBER'

# Twilio credentials for thread 3 (Flood detection)
thread_3_account_sid = 'YOUR_ACCOUNT_SID'
thread_3_auth_token = 'YOUR_AUTH_TOKEN'
thread_3_twilio_number = 'YOUR_TWILIO_NUMBER'
thread_3_recipient_number = 'RECIPIENT_NUMBER'
-------------------------------------------------------------------------------
Run the Script:
python main.py
-------------------------------------------------------------------------------
Functions
send_twilio_alert(account_sid, auth_token, twilio_number, recipient_number, message)
Sends an SMS alert using the Twilio API.

thread_1(video_path, account_sid, auth_token, twilio_number, recipient_number)
Accident detection using YOLO and OpenCV.

thread_2(video_path, account_sid, auth_token, twilio_number, recipient_number)
Wild animal detection using YOLO and OpenCV.

thread_3(video_path, account_sid, auth_token, twilio_number, recipient_number)
Flood detection using YOLO and OpenCV.

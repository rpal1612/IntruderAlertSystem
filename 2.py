import torch
import os
import cv2
import smtplib
from email.message import EmailMessage
import time
import pathlib
import numpy as np
from pathlib import Path

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "ishaandhuri4@gmail.com"
SENDER_PASSWORD = "heqo paom rhta vfbq"
RECIPIENT_EMAIL = "ishaandhuri4@gmail.com"

# Function to send email with attachment
def send_email_with_attachment(file_path, subject="Human Detected!", body="A person was detected in the video. See the attached image."):
    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.set_content(body)

    # Attach file
    with open(file_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)
    msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)

    # Send email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        print(f"Email sent to {RECIPIENT_EMAIL} with attachment: {file_name}")

# New function to detect motion
def detect_motion(frame1, frame2, threshold=20):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(gray1, gray2)

    # Apply threshold to difference
    _, threshold_diff = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

    # Calculate percentage of changed pixels
    changed_pixels = np.sum(threshold_diff > 0)
    total_pixels = threshold_diff.size
    change_percent = (changed_pixels / total_pixels) * 100

    return change_percent > 0.5

# Fix for pathlib issue
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Cache model loading
def load_model(model_path):
    # Check if model is already downloaded
    cache_dir = Path('models_cache')
    cache_dir.mkdir(exist_ok=True)

    try:
        # Load model with caching enabled
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                             path=model_path,
                             force_reload=False,  # Disable force reload
                             verbose=False)  # Reduce printing

        # Configure model for better detection
        model.conf = 0.35  # Lower confidence threshold
        model.iou = 0.45  # IOU threshold
        model.classes = [0]  # Only detect persons
        model.max_det = 50  # Maximum detections per frame

        # Enable GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Set model to evaluation mode
        model.eval()
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model
model_path = os.path.normpath(r'C:\Users\dell\OneDrive\Desktop\best.pt')
model = load_model(model_path)

if model is None:
    print("Failed to load model. Exiting...")
    exit()

# Load video with optimized settings
video_path = r'rtsp://pranavn91@gmail.com:Utkarshcctv1!@192.168.1.36:554/stream1'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer size

# Optimize video capture settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Timer for sending email
email_interval = 10  # seconds
last_email_time = 0
last_detection_time = 0
detection_interval = 0.1  # Run detection every 100ms

# Read first frame
_, previous_frame = cap.read()
frame_count = 0
FORCE_DETECTION_INTERVAL = 30

# Pre-allocate frame buffers
if previous_frame is not None:
    frame_height, frame_width = previous_frame.shape[:2]
    previous_frame_small = cv2.resize(previous_frame, (frame_width // 2, frame_height // 2))

while cap.isOpened():
    # Skip frames if processing is falling behind
    if cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
        cap.grab()

    success, current_frame = cap.read()
    frame_count += 1
    current_time = time.time()

    if success:
        # Resize frame once
        current_frame_small = cv2.resize(current_frame, (frame_width // 2, frame_height // 2))

        # Check for motion less frequently
        if frame_count % 2 == 0 and previous_frame_small is not None:
            motion_detected = detect_motion(previous_frame_small, current_frame_small, threshold=25)  # Lowered threshold
        else:
            motion_detected = False

        # Only run detection if motion detected and enough time has passed
        if (motion_detected or frame_count % FORCE_DETECTION_INTERVAL == 0) and \
           (current_time - last_detection_time) >= detection_interval:

            # Prepare frame for model
            frame_for_model = current_frame.copy()

            # Convert frame to tensor and run inference
            with torch.no_grad():
                results = model(frame_for_model, size=640)  # Fixed size inference

            # Process detections with adjusted confidence
            detections = results.xyxy[0].cpu().numpy()
            high_confidence_detections = detections[
                (detections[:, 4] > 0.35) & (detections[:, 5] == 0)  # Lowered confidence threshold
            ]

            if len(high_confidence_detections) > 0:
                print(f"Detected {len(high_confidence_detections)} persons")  # Debug info

                if current_time - last_email_time >= email_interval:
                    output_image_path = "detected_person.jpg"
                    # Save frame with detections
                    frame_to_save = frame_for_model.copy()
                    for det in high_confidence_detections:
                        x1, y1, x2, y2 = map(int, det[:4])
                        cv2.rectangle(frame_to_save, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.imwrite(output_image_path, frame_to_save)
                    send_email_with_attachment(output_image_path)
                    last_email_time = current_time

                # Render detections
                for det in high_confidence_detections:
                    x1, y1, x2, y2 = map(int, det[:4])
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Person {det[4]:.2f}"
                    cv2.putText(current_frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            last_detection_time = current_time

        # Display the frame
        cv2.imshow('YOLOv5 Detection', current_frame)

        # Update previous frame
        previous_frame_small = current_frame_small

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, previous_frame = cap.read()
        if previous_frame is not None:
            previous_frame_small = cv2.resize(previous_frame, (frame_width // 2, frame_height // 2))
        frame_count = 0

# Release resources
cap.release()
cv2.destroyAllWindows()

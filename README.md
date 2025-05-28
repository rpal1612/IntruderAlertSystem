# Intruder Alert System
An AI-powered Intruder Alert System developed as part of our internship at Utkarsh Minds. This system uses Python and Machine Learning to detect unauthorized access in real-time via video surveillance and trigger alerts instantly.

# Features
- Real-time video stream analysis

- Motion detection using ML models

- Human pose detection (optional)

- Instant alert via buzzer/sound/email/notification

- Logs entry data with timestamps

- Smart monitoring system for homes, offices, and restricted zones

# Tech Stack
**Language**: Python

**Libraries**: OpenCV, NumPy, TensorFlow/Keras or Scikit-learn

**Hardware (optional)**: Raspberry Pi / Webcam

**IDE**: VS Code / Jupyter Notebook

**Platform**: Windows/Linux

# Architecture
![graphviz](https://github.com/user-attachments/assets/136a1f5c-c5f0-46ae-bad7-ca257891942f)


# Installation
    ```bash
    
    git clone https://github.com/your-username/intruder-alert-system.git
    cd intruder-alert-system
    pip install -r requirements.txt
    python main.py
    
# How it Works
- The system captures frames from a webcam or video input.

- It detects motion using frame differencing or background subtraction.

- If motion is detected, the frame is passed through a trained ML model to confirm human presence.

- If confirmed, an alert is triggered through a chosen output (buzzer, email, etc.).

# Applications
- Home security systems

- Bank/ATM surveillance

- Office building access monitoring

- Smart door security systems

# Team Members
Riya Pal

Mangesh Shah

Ishaan Dhuri

Srushti Jadhav

Mayur Parab

# Internship
Developed during our internship at Utkarsh Minds to solve real-world security challenges using AI.

# Screenshot

![WhatsApp Image 2025-04-27 at 20 27 31_1e58333e](https://github.com/user-attachments/assets/c7a8395d-6467-4721-b038-99176e6ad6bb)

# License
This project is licensed under the MIT License - see the LICENSE file for details.

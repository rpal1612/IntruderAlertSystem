# Import the required libraries
import cv2
import time

# Set the RTSP URL with authentication
# Format: rtsp://<username>:<password>@<IP>:<port>/<stream>
rtsp_url = "rtsp://pranavn91@gmail.com:Utkarshcctv1!@192.168.1.36:554/stream1"

# Set window normal so we can resize it
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# Note the starting time
start_time = time.time()

# Initialize these variables for calculating FPS
fps = 0
frame_counter = 0

# Attempt to open the video stream from the camera
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open video stream. Check RTSP URL and credentials.")
    exit()

print("Video stream opened successfully!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame. The stream may have been interrupted.")
        break

    # Calculate the average FPS
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))

    # Display the FPS on the video frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        print("Exiting the stream.")
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

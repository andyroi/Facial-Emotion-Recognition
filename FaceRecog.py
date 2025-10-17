import warnings
import sounddevice as sd
import numpy as np
import cv2
import time
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load Haar cascade classifiers for face, eyes, and smile
# These are used for traditional face, eye, and smile detection
# The XML files are included with OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# Add mouth cascade for detecting open mouth
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')  # We can use smile cascade for open mouth

cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
# Initialize variables for emotion tracking
last_analysis_time = 0
EMOTION_ANALYSIS_INTERVAL = 0.5  # Analyze emotions every 0.5 seconds
emotion_buffer = []
BUFFER_SIZE = 3  # Number of frames to average emotions over

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Only analyze emotions periodically to improve performance
    current_time = time.time()
    should_analyze = current_time - last_analysis_time > EMOTION_ANALYSIS_INTERVAL
    # Convert frame to grayscale for Haar cascades
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    smile_detected = False
    shocked_detected = False
    
    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the face region - more sensitive to wide-open eyes
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.67,  # how sensitive the detection is to size changes
            minNeighbors=12,    # Even fewer neighbors for more detections
            minSize=(25, 1)   # Smaller minimum size to detect wider eyes
        )
        
        # Calculate eye area for shock detection
        total_eye_area = 0
        eye_heights = []  # Track eye heights for better open-eye detection
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            total_eye_area += (ew * eh)
            eye_heights.append(eh)  # Store eye height for ratio calculation

        # Detect open mouth with parameters tuned for larger openings
        mouths = mouth_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,    # More fine-grained detection
            minNeighbors=1,    # Reduced for more detections
            minSize=(50, 30)    # Width > height for open mouth
        )
        
        # Calculate mouth area and position
        mouth_area = 0
        for (mx, my, mw, mh) in mouths:
            # Only consider mouth detections in lower half of face
            if my > h/1.5:
                cv2.rectangle(roi_color, (mx, my), (mx + mw, mh + my), (0, 0, 255), 2)
                mouth_area = mw * mh

        # Detect shocked expression with adjusted thresholds:
        # 1. Larger eye area threshold
        # 2. Larger mouth opening requirement
        # 3. Consider eye height/width ratio
        face_area = w * h
        
        # Calculate average eye height/width ratio (when eyes are wide open, they're more circular)
        eye_ratio = 0
        if len(eye_heights) >= 2:
            eye_ratio = np.mean(eye_heights) / (w * 0.15)  # Compare to expected eye width
        
        # Debug info - print the measurements
        if len(eyes) >= 2 and len(mouths) > 1:
            print(f"\rEye area: {total_eye_area/face_area:.3f}, Mouth area: {mouth_area/face_area:.3f}, Eye ratio: {eye_ratio:.2f}", end='')
        
        if (len(eyes) >= 2 and  # Both eyes detected
            total_eye_area > 0.025 * face_area and  # Eyes are wide (reduced from 0.03)
            mouth_area > 0.10 * face_area and  # Mouth is more open (increased from 0.04)
            eye_ratio > 0.5 and  # Eyes are more circular (indicating wide eyes)
            any(my > h/2 for mx, my, mw, mh in mouths)):  # Mouth in lower face
            shocked_detected = True

        # Regular smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
        if len(smiles) > 0 and not shocked_detected:  # Don't detect smile if shocked
            smile_detected = True
        for (sx, sy, sw, sh) in smiles:
            if not shocked_detected:  # Only draw smile if not shocked
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    # If a smile is detected, overlay 'Smiled!' text
    image_path_smile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smile.png')
    image_path_goon = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'goon.png')
    yay_img = None
    goon_img = None

    if os.path.exists(image_path_smile):
        yay_img = cv2.imread(image_path_smile, cv2.IMREAD_UNCHANGED)
        goon_img = cv2.imread(image_path_goon, cv2.IMREAD_UNCHANGED)

    # Handle shocked expression first (priority over smile)
    if shocked_detected: # Display goon image
        if goon_img is not None:
            # Resize the image to fit the frame
            goon_img = cv2.resize(goon_img, (frame.shape[1], frame.shape[0]))
            # Overlay the image on the frame
            frame = cv2.addWeighted(frame, 0.5, goon_img, 0.5, 0)
        else:
            # Fallback to text if image is not available
            cv2.putText(frame, 'Smiled!', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
    elif smile_detected: # Handle smile if not shocked
        if yay_img is not None:
            # Resize the image to fit the frame
            yay_img = cv2.resize(yay_img, (frame.shape[1], frame.shape[0]))
            # Overlay the image on the frame
            frame = cv2.addWeighted(frame, 0.5, yay_img, 0.5, 0)
        else:
            # Fallback to text if image is not available
            cv2.putText(frame, 'Smiled!', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)


    # Show the frame after all overlays
    cv2.imshow('Webcam - Face, Eye & Smile Detection', frame)
    # Wait for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
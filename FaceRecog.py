import warnings
import sounddevice as sd
import numpy as np
import cv2
from deepface import DeepFace
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load Haar cascade classifiers for face, eyes, and smile
# These are used for traditional face, eye, and smile detection
# The XML files are included with OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

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
    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes: # creation of rectangle around eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect smiles in the face region
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
        if len(smiles) > 0:
            smile_detected = True
        for (sx, sy, sw, sh) in smiles: # creation of rectangle around smile
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    # If a smile is detected, overlay 'Smiled!' text
    yay_img = cv2.imread('IMG_5180.jpg', cv2.IMREAD_UNCHANGED)
    if smile_detected:
        # Resize the sad image to fit the frame
        yay_img = cv2.resize(yay_img, (frame.shape[1], frame.shape[0]))
        # Overlay the sad image on the frame
        frame = cv2.addWeighted(frame, 0.5, yay_img, 0.5, 0)
        #cv2.putText(frame, 'Smiled!', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)


    # Emotion detection using DeepFace
    sad_detected = False
    
    if should_analyze and len(faces) > 0:
        try:
            # Analyze emotions using DeepFace
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Update last analysis time
            last_analysis_time = current_time
            
            # Get emotion predictions
            if isinstance(result, list):
                result = result[0]  # Take first face if multiple detected
                
            emotions = result['emotion']
            # print("Emotion scores:", emotions)
            
            # Add current emotions to buffer
            emotion_buffer.append(emotions)
            if len(emotion_buffer) > BUFFER_SIZE:
                emotion_buffer.pop(0)
            
            # Calculate average emotions over buffer
            avg_emotions = {}
            for emotion in ['sad', 'neutral', 'happy', 'surprise']:
                avg_emotions[emotion] = np.mean([frame[emotion] for frame in emotion_buffer])
            
            # Enhanced emotion detection logic
            sad_score = avg_emotions['sad']
            neutral_score = avg_emotions['neutral']
            happy_score = avg_emotions['happy']
            
            # More sophisticated emotion analysis:
            # 1. High neutral threshold (45%)
            # 2. Requires sustained sad expression
            # 3. Considers happiness as direct counterweight
            print(f"Sustained sad expression detected: sad={sad_score:.1f}%, "
                f"neutral={neutral_score:.1f}%, happy={happy_score:.1f}%")
            if neutral_score > 45:  # Strong neutral detection
                sad_detected = False
            elif (sad_score > 60 and  # Need 60%+ confidence
                  sad_score > neutral_score + 20 and  # Must be clearly sadder than neutral
                  sad_score > happy_score + 30):  # Must be much sadder than happy
                sad_detected = True
                print(f"Sustained sad expression detected: sad={sad_score:.1f}%, "
                      f"neutral={neutral_score:.1f}%, happy={happy_score:.1f}%")
                
        except Exception as e:
            print(f"Emotion analysis error: {str(e)}")
            # Clear emotion buffer on error
            emotion_buffer.clear()

    if sad_detected:
        cv2.putText(frame, 'Sad!', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)


    # Show the frame after all overlays
    cv2.imshow('Webcam - Face, Eye & Smile Detection', frame)
    # Wait for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
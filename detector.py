import cv2
import numpy as np
from PIL import Image
import os

# Function to detect faces, recognize them, and display results on the webcam feed
def rectangle(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    
    # Loop through all detected faces
    for (x, y, w, h) in features:
        # Draw a rectangle around each detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Predict the ID of the detected face using the trained model
        id, pred = clf.predict(gray[y:y + h, x:x + w])
        # Calculate the confidence level (higher means more confidence)
        confidence = int(100 * (1 - pred / 300))
        
        # If the confidence is above 75%, display the recognized person's name
        if confidence > 75:
            if id == 1:
                cv2.putText(img, "PERSON NAME", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            if id == 2:
                cv2.putText(img, "PERSON NAME", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            # If the confidence is below 75%, display "ALIEN" (or unknown face)
            cv2.putText(img, "ALIEN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    return img

# Load the pre-trained Haar Cascade face detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained face recognition model
detect = cv2.face.LBPHFaceRecognizer_create()
detect.read("detector.xml")

# Start the webcam to capture live video
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam feed
    ret, img = video_capture.read()

    # Flip the image horizontally to make it look more natural (non-mirrored)
    img = cv2.flip(img, 1)
    
    # Call the rectangle function to detect faces and recognize them
    img = rectangle(img, faceCascade, 1.3, 6, (500, 0, 0), "Face", detect)
    
    # Show the processed image with detected faces and predictions
    cv2.imshow("face Detector", img)
    
    # Stop the webcam feed when 'Enter' key is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the webcam and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

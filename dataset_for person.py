import cv2

# Function to generate the dataset by capturing face images from webcam
def generate_dataset():
    # Load OpenCV's pre-trained Haar Cascade face detector
    face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Function to detect and crop faces from the input image
    def face_cropped(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        faces = face_detect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the image

        # If no faces are detected, return None
        if faces == ():
            return None
        # Crop the detected face from the image
        for (x, y, w, h) in faces:
            crop_face = image[y:y + h, x:x + w]
        return crop_face

    # Start the webcam to capture live video feed
    video_cap = cv2.VideoCapture(0)
    
    # ID for the user (can be changed for different people)
    id = 1
    img_id = 0  # Initialize the image ID for saving images

    while True:
        # Capture each frame from the webcam feed
        ret, frame = video_cap.read()
        
        # Check if a face is detected in the frame
        if face_cropped(frame) is not None:
            img_id += 1  # Increment image ID for each face captured
            face = cv2.resize(face_cropped(frame), (200, 200))  # Resize the cropped face
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert the face to grayscale
            # Save the image in the 'dataset' folder with a unique name
            file_name_path = "dataset/user." + str(id) + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)  # Save the face image
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Display image ID on the face
            
            # Display the cropped face image in a window
            cv2.imshow("Cropped face", face)

        # Stop capturing when 'Enter' key is pressed or when 1000 images are captured
        if cv2.waitKey(1) == 13 or int(img_id) == 1000:
            break

    video_cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("APKA IMAGE CAPTURE HO CHUKA HA KRIPYA APNA THOBDA HATA LA ._.")  # Message indicating capture is done

# Run the function to start capturing face images
generate_dataset()

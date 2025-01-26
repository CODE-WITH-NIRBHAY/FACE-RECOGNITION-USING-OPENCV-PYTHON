import os
import cv2
from PIL import Image
import numpy as np

# Function to train a face recognition model using the captured dataset
def train_detector(dataset_dir):
    # Get all image file paths from the dataset directory
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    
    faces = []  # List to store cropped face images
    ids = []  # List to store corresponding user IDs

    # Loop through all images in the dataset
    for image in path:
        # Open the image and convert it to grayscale
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')  # Convert the image to a NumPy array
        # Extract the user ID from the filename (assumes filename format 'user.ID.number.jpg')
        id = int(os.path.split(image)[1].split(".")[1])
        
        faces.append(imageNp)  # Append the image to the faces list
        ids.append(id)  # Append the corresponding ID to the ids list

    ids = np.array(ids)  # Convert the list of IDs to a NumPy array

    # Initialize the face recognizer (LBPH method)
    detect = cv2.face.LBPHFaceRecognizer_create()
    # Train the model using the faces and ids
    detect.train(faces, ids)
    # Save the trained model to a file
    detect.write("detector.xml")

# Run the function to start training the model with the captured dataset
train_detector("dataset")

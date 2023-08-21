import dlib
import numpy as np
import cv2
import os

# Load pre-trained facial shape predictor and face recognition model from dlib
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Initialize the dlib face detector
detector = dlib.get_frontal_face_detector()

# Function to extract facial features from an image
def extract_face_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = detector(gray_image)
    
    if len(detected_faces) == 0:
        return []
    
    face_descriptors = []
    for face in detected_faces:
        shape = shape_predictor(gray_image, face)
        face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
        face_descriptors.append(face_descriptor)
    
    return face_descriptors

    
input_image = cv2.imread('10.jpg')  # Load the input image
input_face_features = extract_face_features(input_image)
# Load and preprocess your array of images
array_of_images = ['2.jpeg', '3.jpeg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg','9.jpg','10.jpg','11.jpg','12.jpg']

# Store images containing the input face
matching_images = []

for image_path in array_of_images:
    image = cv2.imread(image_path)
    image_face_features = extract_face_features(image)
    
    for image_features in image_face_features:
        if len(input_face_features) == 0 or len(image_face_features) == 0:
            continue
        
        distance = np.linalg.norm(np.array(image_features) - np.array(input_face_features[0]))
        threshold = 0.6  # Adjust as needed
        
        if distance < threshold:
            matching_images.append(image_path)  # Keep filename with extension
            break  # Move to the next image once a match is found

# Display or further process the matching images
# for image_name in matching_images:
#     print(image_name)
print(matching_images)

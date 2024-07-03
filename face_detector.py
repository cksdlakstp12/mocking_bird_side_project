import cv2
import numpy as np

def extract_faces(image_path, output_path):
    # Load the cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop through the faces detected
    for i, (x, y, w, h) in enumerate(faces):
        # Extract the face
        face = img[y:y+h, x:x+w]

        # Save the face image
        face_output_path = f"{output_path}/face_{i+1}.jpg"
        cv2.imwrite(face_output_path, face)

        print(f"Face {i+1} saved at {face_output_path}")

# Example usage
image_path = 'input.jpg'  # Replace with your image path
output_path = '.'  # Replace with your output directory
extract_faces(image_path, output_path)

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os

class ImageFaceExtractor:
    def __init__(self, root, image_folder, save_folder):
        self.root = root
        self.root.title("Image Face Extractor")
        self.image_folder = image_folder
        self.save_folder = save_folder
        self.images = os.listdir(self.image_folder)
        self.current_index = 0
        self.faces = []
        self.face_index = 0
        
        # Load the cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.label_display = tk.Label(root)
        self.label_display.pack(pady=10)
        
        self.info_label = tk.Label(root, text="Press 'a' to save face, 's' to skip face")
        self.info_label.pack()
        
        # Bind keyboard events
        self.root.bind('<KeyPress-a>', self.save_face)
        self.root.bind('<KeyPress-s>', self.skip_face)
        
        self.show_image_and_extract_faces()

    def show_image_and_extract_faces(self):
        image_path = os.path.join(self.image_folder, self.images[self.current_index])
        
        # Read the input image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # Reset previous faces
        self.faces = []

        # Loop through the faces detected
        for (x, y, w, h) in faces:
            # Extract the face
            face = img[y:y+h, x:x+w]
            self.faces.append(face)

            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Convert image to RGB format for tkinter display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.label_display.config(image=img_tk)
        self.label_display.image = img_tk  # Keep a reference to avoid garbage collection
        self.root.title(f"Image {self.current_index + 1}/{len(self.images)} - Faces Detected: {len(self.faces)}")

    def save_face(self, event):
        if self.faces:
            # Save the face image
            face_output_path = os.path.join(self.save_folder, f"{self.face_index + 1}.jpg")
            cv2.imwrite(face_output_path, self.faces[self.face_index])
            self.face_index += 1
            self.show_next_face()

    def skip_face(self, event):
        self.show_next_face()

    def show_next_face(self):
        self.face_index += 1
        if self.face_index < len(self.faces):
            self.show_image_and_extract_faces()
        else:
            self.next_image()

    def next_image(self):
        self.current_index += 1
        self.face_index = 0
        if self.current_index < len(self.images):
            self.show_image_and_extract_faces()
        else:
            self.label_display.config(text="No more images to process.")
            self.info_label.config(text="")

# Example usage:
if __name__ == "__main__":
    image_folder = "/path/to/your/image/folder"
    save_folder = "/path/to/your/save/folder"
    
    root = tk.Tk()
    face_extractor = ImageFaceExtractor(root, image_folder, save_folder)
    root.mainloop()

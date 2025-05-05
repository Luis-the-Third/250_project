import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import warnings
import time

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_current_frame = True
        self.capture_mode = False
        self.name_to_capture = ""

    def load_known_faces(self, known_faces_dir):
        """Load known faces from a directory"""
        try:
            for image_name in os.listdir(known_faces_dir):
                if image_name.startswith('.'):
                    continue  # skip hidden files
                
                image_path = os.path.join(known_faces_dir, image_name)
                face_image = face_recognition.load_image_file(image_path)
                
                # Get all face encodings (there might be multiple in one image)
                encodings = face_recognition.face_encodings(face_image)
                if encodings:  # If at least one face found
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(os.path.splitext(image_name)[0])
            
            print(f"Loaded {len(self.known_face_names)} known faces")
        except Exception as e:
            print(f"Error loading known faces: {str(e)}")

    def recognize_faces(self, frame):
        """Recognize faces in a video frame"""
        if self.capture_mode:
            cv2.putText(frame, "Capturing...", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

        if self.process_current_frame:
            # Find face locations using both models
            self.face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            if not self.face_locations:  # Fallback to CNN if HOG fails
                self.face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            
            if self.face_locations:
                try:
                    self.face_encodings = face_recognition.face_encodings(
                        rgb_small_frame,
                        known_face_locations=self.face_locations,
                        num_jitters=1
                    )
                except Exception as e:
                    print(f"Encoding error: {str(e)}")
                    self.face_encodings = []
                
                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "0%"
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = f"{100 - face_distances[best_match_index] * 100:.2f}%"
                    
                    self.face_names.append(f"{name} ({confidence})")
            else:
                self.face_encodings = []
                self.face_names = []

        self.process_current_frame = not self.process_current_frame

        # Display results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)

        return frame

    def start_capture(self, name):
        """Prepare to capture a new face"""
        self.capture_mode = True
        self.name_to_capture = name

    def capture_new_face(self, frame, name):
        """Capture and save a new face"""
        try:
            rgb_frame = frame[:, :, ::-1]  # Convert to RGB
            
            # Try multiple detection methods
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            if not face_locations:
                face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
            
            if len(face_locations) != 1:
                print(f"Error: Need exactly 1 face, found {len(face_locations)}")
                return False
                
            # Get encodings
            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                known_face_locations=face_locations,
                num_jitters=1
            )
            
            if not face_encodings:
                print("Error: Could not encode face")
                return False
                
            # Save the new face
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved new face: {filename}")
            return True
            
        except Exception as e:
            print(f"Capture error: {str(e)}")
            return False

def main():
    system = FaceRecognitionSystem()
    
    # Load known faces
    known_faces_dir = "known_faces"
    if os.path.exists(known_faces_dir):
        system.load_known_faces(known_faces_dir)
    else:
        os.makedirs(known_faces_dir, exist_ok=True)
        print(f"Created directory: {known_faces_dir}")

    # Initialize camera
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow('Face Recognition')
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame")
            break

        processed_frame = system.recognize_faces(frame)
        cv2.imshow('Face Recognition', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            video_capture.release()
            cv2.destroyAllWindows()
            name = input("Enter name for new face: ").strip()
            if name:  # Only proceed if name was entered
                video_capture = cv2.VideoCapture(0)
                cv2.namedWindow('Face Recognition')
                system.start_capture(name)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
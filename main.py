import cv2
import numpy as np
import os
import time
import easygui
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
import pickle

class SimpleFaceRecognition:
    def __init__(self, id_folder, threshold=0.6):
        self.threshold = threshold
        self.id_folder = id_folder
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if self.face_cascade.empty():
            raise Exception("Could not load face detection cascade classifier")
        
        os.makedirs(id_folder, exist_ok=True)
        
        # Initialize face recognizer components
        self.face_recognizer = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        
        self.load_and_train_identities()
    
    def ensure_uint8(self, image):
        """Ensure image is in uint8 format"""
        if image.dtype != np.uint8:
            # Convert to uint8
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image
    
    def extract_face_features(self, face_image):
        """Extract simple features from face using histogram and HOG-like features"""
        # Ensure image is uint8
        face_image = self.ensure_uint8(face_image)
        
        # Resize to standard size
        face_resized = cv2.resize(face_image, (100, 100))
        
        # Convert to grayscale if needed
        if len(face_resized.shape) == 3:
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_resized
        
        # Simple feature extraction: histogram
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])  # Reduced bins for efficiency
        hist = cv2.normalize(hist, hist).flatten()
        
        # Add some simple texture features using Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_features = np.mean(gradient_magnitude), np.std(gradient_magnitude)
        
        # Combine features
        features = np.concatenate([
            hist,
            np.array(gradient_features),
            gray.flatten() / 255.0  # Normalized pixel values
        ])
        
        return features
    
    def load_and_train_identities(self):
        print("Loading known identities and training recognizer...")
        features = []
        labels = []
        label_id = 0
        
        try:
            person_folders = os.listdir(os.path.expanduser(self.id_folder))
        except FileNotFoundError:
            print(f"ID folder {self.id_folder} not found")
            return
        
        if not person_folders:
            print("No person folders found")
            return
        
        for person_name in person_folders:
            person_dir = os.path.join(self.id_folder, person_name)
            if os.path.isdir(person_dir):
                print(f"Processing {person_name}...")
                self.label_encoder[label_id] = person_name
                self.reverse_label_encoder[person_name] = label_id
                
                image_count = 0
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, img_file)
                        face_images = self.extract_faces_from_image(img_path)
                        
                        for face in face_images:
                            try:
                                face_features = self.extract_face_features(face)
                                features.append(face_features)
                                labels.append(label_id)
                                image_count += 1
                            except Exception as e:
                                print(f"Error processing face in {img_file}: {e}")
                                continue
                
                print(f"  Found {image_count} face images for {person_name}")
                label_id += 1
        
        if features and labels:
            # Train KNN classifier
            self.face_recognizer = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
            self.face_recognizer.fit(features, labels)
            print(f"Trained recognizer with {len(features)} faces for {label_id} people")
        else:
            print("No face images found for training")
            self.face_recognizer = None
    
    def extract_faces_from_image(self, image_path):
        """Extract faces from a single image file"""
        faces = []
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return faces
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in detected_faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            # Ensure minimum size
            if w > 50 and h > 50:
                faces.append(face_roi)
        
        return faces
    
    def detect_faces(self, img):
        if img is None or img.size == 0:
            return [], [], []
            
        margin = 44
        image_size = 160
        
        # Ensure input is uint8
        img = self.ensure_uint8(img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_patches = []
        bounding_boxes = []
        landmarks = []
        
        for (x, y, w, h) in faces:
            # Expand bounding box
            bb = [
                max(x - margin//2, 0),
                max(y - margin//2, 0),
                min(x + w + margin//2, img.shape[1]),
                min(y + h + margin//2, img.shape[0])
            ]
            
            if bb[2] <= bb[0] or bb[3] <= bb[1]:
                continue
                
            try:
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                if cropped.size == 0:
                    continue
                    
                aligned = cv2.resize(cropped, (image_size, image_size))
                # Skip prewhitening to avoid float conversion issues
                # prewhitened = self.prewhiten(aligned)
                
                face_patches.append(aligned)  # Use aligned directly instead of prewhitened
                bounding_boxes.append(bb)
                landmarks.append([bb[0], bb[1], bb[2], bb[3], bb[0], 
                                bb[1], bb[2], bb[3], bb[0], bb[1]])
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        return face_patches, bounding_boxes, landmarks
    
    def prewhiten(self, x):
        """Optional prewhitening - removed to avoid float conversion issues"""
        return x
    
    def generate_embeddings(self, face_patches):
        """Convert face patches to feature vectors"""
        features = []
        for face in face_patches:
            try:
                # Ensure face is in correct format
                face_uint8 = self.ensure_uint8(face)
                face_features = self.extract_face_features(face_uint8)
                features.append(face_features)
            except Exception as e:
                print(f"Error generating features: {e}")
                # Add dummy features if error occurs
                dummy_features = np.zeros(100 * 100 + 66)  # Match feature dimension
                features.append(dummy_features)
        
        return np.array(features)
    
    def find_matching_ids(self, face_features):
        """Recognize faces using the trained classifier"""
        matching_ids = []
        matching_distances = []
        
        if self.face_recognizer is None or len(face_features) == 0:
            return [None] * len(face_features), [None] * len(face_features)
        
        try:
            # Find nearest neighbors
            distances, indices = self.face_recognizer.kneighbors(face_features)
            
            for i, (distance, index) in enumerate(zip(distances, indices)):
                min_distance = distance[0]
                label_index = index[0]
                
                if min_distance < self.threshold:
                    person_name = self.label_encoder.get(label_index, "Unknown")
                    matching_ids.append(person_name)
                    matching_distances.append(min_distance)
                else:
                    matching_ids.append(None)
                    matching_distances.append(min_distance)
                    
        except Exception as e:
            print(f"Error in face recognition: {e}")
            matching_ids = [None] * len(face_features)
            matching_distances = [None] * len(face_features)
        
        return matching_ids, matching_distances
    
    def add_identity(self, face_patch, name):
        """Add a new identity to the system"""
        if not name or not name.strip():
            print("Invalid name provided")
            return False
            
        try:
            # Ensure face patch is uint8
            face_patch = self.ensure_uint8(face_patch)
            
            # Save the face image
            id_dir = os.path.join(self.id_folder, name)
            os.makedirs(id_dir, exist_ok=True)
            
            existing_files = [f for f in os.listdir(id_dir) if f.endswith('.jpg')]
            next_num = len(existing_files)
            
            cv2.imwrite(os.path.join(id_dir, f"{next_num}.jpg"), face_patch)
            print(f"Saved face image for: {name}")
            
            # Retrain the recognizer with the new identity
            self.load_and_train_identities()
            
            print(f"Successfully added and retrained for: {name}")
            return True
            
        except Exception as e:
            print(f"Error adding identity: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Face Recognition System')
    parser.add_argument('id_folder', type=str, help='Folder containing ID folders')
    parser.add_argument('-t', '--threshold', type=float, default=0.6, help='Distance threshold for matching (lower = stricter)')
    
    args = parser.parse_args()
    
    try:
        face_rec = SimpleFaceRecognition(args.id_folder, args.threshold)
    except Exception as e:
        print(f"Failed to initialize face recognition: {e}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Face Recognition System Started!")
    print("Hotkeys:")
    print("  l - Toggle landmarks")
    print("  b - Toggle bounding boxes") 
    print("  i - Toggle ID display")
    print("  f - Toggle FPS display")
    print("  s - Save current face as new identity")
    print("  q - Quit")
    print(f"Distance threshold: {args.threshold} (lower = better match)")
    
    show_landmarks = False
    show_bb = True
    show_id = True
    show_fps = False
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            face_patches, bounding_boxes, landmarks = face_rec.detect_faces(frame)
            
            if len(face_patches) > 0:
                face_features = face_rec.generate_embeddings(face_patches)
                matching_ids, matching_distances = face_rec.find_matching_ids(face_features)
                
                print(f"Detected {len(face_patches)} face(s):")
                
                for i, (bb, landmark, match_id, dist) in enumerate(zip(bounding_boxes, landmarks, matching_ids, matching_distances)):
                    if match_id is None:
                        display_name = "Unknown"
                        color = (0, 0, 255)
                        if dist is not None:
                            print(f"  Face {i+1}: Unknown (distance: {dist:.4f})")
                        else:
                            print(f"  Face {i+1}: Unknown")
                    else:
                        display_name = match_id
                        color = (0, 255, 0)
                        if dist is not None:
                            print(f"  Face {i+1}: {match_id} (distance: {dist:.4f})")
                        else:
                            print(f"  Face {i+1}: {match_id}")
                    
                    if show_bb:
                        cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    
                    if show_id:
                        text = f"{display_name}"
                        if dist is not None:
                            text += f" ({dist:.3f})"
                        cv2.putText(frame, text, (bb[0], bb[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if show_landmarks:
                        for j in range(5):
                            x = int(landmark[j])
                            y = int(landmark[j+5])
                            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
            else:
                print("No faces detected")
            
            fps = 1.0 / (time.time() - start_time)
            if show_fps:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                show_landmarks = not show_landmarks
                print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
            elif key == ord('b'):
                show_bb = not show_bb
                print(f"Bounding boxes: {'ON' if show_bb else 'OFF'}")
            elif key == ord('i'):
                show_id = not show_id
                print(f"ID display: {'ON' if show_id else 'OFF'}")
            elif key == ord('f'):
                show_fps = not show_fps
                print(f"FPS display: {'ON' if show_fps else 'OFF'}")
            elif key == ord('s') and len(bounding_boxes) > 0:
                bb = bounding_boxes[0]
                face_patch = frame[bb[1]:bb[3], bb[0]:bb[2]]
                
                cv2.imshow('Captured Face', face_patch)
                cv2.waitKey(100)
                
                name = easygui.enterbox("Enter name for this person:")
                if name and name.strip():
                    if face_rec.add_identity(face_patch, name.strip()):
                        print(f"Successfully added: {name}")
                    else:
                        print(f"Failed to add: {name}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
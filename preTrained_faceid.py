import os
import cv2
import numpy as np
import torch
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories
os.makedirs("faces", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Initialize models
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, device=device
)

# Load pre-trained FaceNet model (VGGFace2)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Dictionary to store registered face embeddings
registered_faces = {}

# Load existing embeddings if available
embedding_path = 'models/face_embeddings.pkl'
if os.path.exists(embedding_path):
    try:
        with open(embedding_path, 'rb') as f:
            registered_faces = pickle.load(f)
        print(f"Loaded {len(registered_faces)} registered faces")
    except Exception as e:
        print(f"Error loading embeddings: {e}")

def get_embedding(face_img):
    """Get embedding from face image"""
    # Convert to RGB if needed
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    elif face_img.shape[2] == 4:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
    elif face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    face_img = Image.fromarray(face_img)
    
    # Get face embedding
    face_tensor = mtcnn(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(face_tensor)
    
    return embedding.cpu().numpy()[0]

def register_face(name):
    """Register a new face"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    embeddings = []
    count = 0
    max_samples = 5  # Collect multiple samples for better accuracy
    
    print(f"Registering {name}. Please look at the camera...")
    print(f"We'll take {max_samples} samples. Press 'c' to capture each sample.")
    
    while count < max_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Display the frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Sample {count+1}/{max_samples} - Press 'c' to capture", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                box = [int(b) for b in box]
                cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        cv2.imshow('Register Face', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Try to detect face
            if boxes is not None and len(boxes) > 0:
                # Get largest face
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                largest_idx = np.argmax(areas)
                box = [int(b) for b in boxes[largest_idx]]
                
                # Extract face using MTCNN for consistent processing
                face_img = frame[box[1]:box[3], box[0]:box[2]]
                
                try:
                    # Get embedding
                    embedding = get_embedding(face_img)
                    embeddings.append(embedding)
                    count += 1
                    
                    # Save face image
                    face_dir = os.path.join("faces", name)
                    os.makedirs(face_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(face_dir, f"sample_{count}.jpg"), face_img)
                    
                    print(f"Sample {count}/{max_samples} captured")
                except Exception as e:
                    print(f"Error processing face: {e}")
            else:
                print("No face detected! Please position your face in the frame.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(embeddings) > 0:
        # Average the embeddings for robustness
        avg_embedding = np.mean(embeddings, axis=0)
        # Normalize the embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # Store in the dictionary
        registered_faces[name] = avg_embedding
        
        # Save to file
        with open(embedding_path, 'wb') as f:
            pickle.dump(registered_faces, f)
        
        print(f"Successfully registered {name}")
        return True
    else:
        print(f"Failed to register {name}")
        return False

def verify_face():
    """Verify a face against registered faces"""
    if not registered_faces:
        print("No faces are registered yet!")
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Looking for faces to verify...")
    print("Press 'q' to quit, 'v' to verify current face")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                box = [int(b) for b in box]
                cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Show instructions
        cv2.putText(display_frame, "Press 'v' to verify, 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                   
        cv2.imshow('Verify Face', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            if boxes is not None and len(boxes) > 0:
                # Get largest face
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                largest_idx = np.argmax(areas)
                box = [int(b) for b in boxes[largest_idx]]
                
                # Extract face
                face_img = frame[box[1]:box[3], box[0]:box[2]]
                
                try:
                    # Get embedding
                    embedding = get_embedding(face_img)
                    
                    # Compare with registered faces
                    best_match = None
                    best_similarity = -1
                    threshold = 0.6  # Adjust as needed, higher = stricter
                    
                    for name, reg_embedding in registered_faces.items():
                        similarity = np.dot(embedding, reg_embedding)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = name
                    
                    # Display result
                    result_frame = frame.copy()
                    
                    if best_similarity > threshold:
                        # Authentication success
                        cv2.putText(result_frame, f"Welcome, {best_match}!", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(result_frame, f"Confidence: {best_similarity:.2f}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print(f"Authentication successful: {best_match} with confidence {best_similarity:.4f}")
                    else:
                        # Authentication failed
                        cv2.putText(result_frame, "Access Denied", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(result_frame, f"Best match: {best_match} ({best_similarity:.2f})", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print(f"Authentication failed. Best match was {best_match} with confidence {best_similarity:.4f}")
                    
                    cv2.imshow("Verification Result", result_frame)
                    cv2.waitKey(3000)  # Display result for 3 seconds
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
            else:
                print("No face detected! Please position your face in the frame.")
    
    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    """Display main menu"""
    while True:
        print("\n==== Face ID System ====")
        print("1. Register a new face")
        print("2. Verify a face")
        print("3. List registered faces")
        print("4. Delete a registration")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            name = input("Enter name to register: ")
            if name.strip():
                register_face(name)
            else:
                print("Name cannot be empty")
                
        elif choice == '2':
            verify_face()
            
        elif choice == '3':
            if not registered_faces:
                print("No faces registered yet")
            else:
                print("\nRegistered faces:")
                for i, name in enumerate(registered_faces.keys(), 1):
                    print(f"{i}. {name}")
                    
        elif choice == '4':
            if not registered_faces:
                print("No faces registered yet")
                continue
                
            print("\nRegistered faces:")
            names = list(registered_faces.keys())
            for i, name in enumerate(names, 1):
                print(f"{i}. {name}")
                
            try:
                idx = int(input("\nEnter number to delete (0 to cancel): ")) - 1
                if idx == -1:  # User entered 0 to cancel
                    print("Deletion canceled")
                    continue
                    
                name_to_delete = names[idx]
                del registered_faces[name_to_delete]
                print(f"Deleted {name_to_delete}")
                
                # Save updated embeddings
                with open(embedding_path, 'wb') as f:
                    pickle.dump(registered_faces, f)
            except (ValueError, IndexError):
                print("Invalid selection")
                
        elif choice == '5':
            print("Exiting program")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()

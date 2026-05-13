import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

def extract_face(frame, face_cascade_path='haarcascade_frontalface_default.xml'):
    """
    Extract face from a frame using Haar Cascade with multiple attempts
    """
    # Download the Haar cascade file if it doesn't exist
    if not os.path.exists(face_cascade_path):
        print("Downloading Haar Cascade file...")
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        urllib.request.urlretrieve(url, face_cascade_path)
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try multiple detection parameters
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # More sensitive
    
    if len(faces) == 0:
        # Try with different parameters
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        # Try even more sensitive parameters
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
    
    if len(faces) > 0:
        # Use the largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Expand the face region slightly
        expand = 20
        x = max(0, x - expand)
        y = max(0, y - expand)
        w = min(frame.shape[1] - x, w + 2 * expand)
        h = min(frame.shape[0] - y, h + 2 * expand)
        
        face = frame[y:y+h, x:x+w]
        return cv2.resize(face, (224, 224))  # Resize to standard size
    
    return None

def extract_face_robust(frame):
    """
    More robust face extraction with multiple fallbacks
    """
    # Try standard face detection first
    face = extract_face(frame)
    
    if face is not None:
        return face
    
    # If no face detected, try to find the largest "face-like" region
    # Convert to different color spaces for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use the entire image as fallback for synthetic faces
    height, width = frame.shape[:2]
    
    # For synthetic images, assume face is in center
    center_x, center_y = width // 2, height // 2
    face_size = min(width, height) // 2
    
    x = center_x - face_size // 2
    y = center_y - face_size // 2
    w = face_size
    h = face_size
    
    # Ensure coordinates are within bounds
    x = max(0, x)
    y = max(0, y)
    w = min(width - x, w)
    h = min(height - y, h)
    
    if w > 50 and h > 50:  # Minimum reasonable face size
        face = frame[y:y+h, x:x+w]
        return cv2.resize(face, (224, 224))
    
    return None

def create_directories():
    """Create necessary directories"""
    os.makedirs('data/real', exist_ok=True)
    os.makedirs('data/fake', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("Directories created successfully!")

def show_sample_images(real_images, fake_images):
    """
    Display sample real and fake images
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        if i < len(real_images):
            axes[0, i].imshow(cv2.cvtColor(real_images[i], cv2.COLOR_BGR2RGB))
            axes[0, i].set_title('Real')
            axes[0, i].axis('off')
        
        if i < len(fake_images):
            axes[1, i].imshow(cv2.cvtColor(fake_images[i], cv2.COLOR_BGR2RGB))
            axes[1, i].set_title('Fake')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
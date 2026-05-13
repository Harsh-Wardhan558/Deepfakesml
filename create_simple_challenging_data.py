import os
import cv2
import numpy as np
from utils import create_directories

def create_simple_challenging_dataset(num_samples=200):
    """Simple version without complex operations"""
    create_directories()
    
    # Clear existing data
    for folder in ['data/real', 'data/fake']:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    os.remove(os.path.join(folder, file))
    
    print("Creating simple challenging dataset...")
    
    for i in range(num_samples):
        # Simple real face
        img_real = np.full((224, 224, 3), 200, dtype=np.uint8)
        cv2.circle(img_real, (112, 112), 60, (255, 220, 180), -1)
        cv2.circle(img_real, (90, 100), 8, (0, 0, 0), -1)
        cv2.circle(img_real, (134, 100), 8, (0, 0, 0), -1)
        cv2.ellipse(img_real, (112, 130), (20, 10), 0, 0, 180, (0, 0, 0), 3)
        cv2.imwrite(f'data/real/real_{i:03d}.jpg', img_real)
        
        # Simple fake face with clear artifacts
        img_fake = np.full((224, 224, 3), 150, dtype=np.uint8)
        cv2.rectangle(img_fake, (62, 62), (162, 162), (220, 200, 170), -1)
        cv2.rectangle(img_fake, (80, 90), (100, 110), (0, 0, 0), -1)
        cv2.rectangle(img_fake, (124, 90), (144, 110), (0, 0, 0), -1)
        cv2.line(img_fake, (90, 140), (134, 140), (0, 0, 0), 3)
        
        # Add clear artifacts
        noise = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
        img_fake = cv2.add(img_fake, noise)
        
        # Add blur to simulate deepfake artifacts
        img_fake = cv2.GaussianBlur(img_fake, (5, 5), 0)
        
        cv2.imwrite(f'data/fake/fake_{i:03d}.jpg', img_fake)
    
    print(f"✅ Simple dataset created with {num_samples} samples each!")
    return True

if __name__ == "__main__":
    create_simple_challenging_dataset(200)
    print("🎉 Ready for training! Run: python train_advanced.py")
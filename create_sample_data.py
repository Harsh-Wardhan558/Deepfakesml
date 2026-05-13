import os
import cv2
import numpy as np
from utils import create_directories

def create_sample_dataset(num_samples=200):
    """Create a better sample dataset for testing"""
    create_directories()
    
    print("Creating improved sample dataset...")
    
    # Create "real" faces (more natural looking)
    for i in range(num_samples):
        # Base image with skin-like color
        img = np.random.randint(180, 220, (224, 224, 3), dtype=np.uint8)
        
        # Draw a realistic face oval
        center_x, center_y = 112, 112
        
        # Face oval
        cv2.ellipse(img, (center_x, center_y), (80, 100), 0, 0, 360, (200, 180, 150), -1)
        
        # Eyes - more realistic
        cv2.ellipse(img, (center_x-35, center_y-15), (15, 8), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(img, (center_x+35, center_y-15), (15, 8), 0, 0, 360, (50, 50, 50), -1)
        
        # Pupils
        cv2.circle(img, (center_x-35, center_y-15), 4, (0, 0, 0), -1)
        cv2.circle(img, (center_x+35, center_y-15), 4, (0, 0, 0), -1)
        
        # Nose
        cv2.line(img, (center_x, center_y), (center_x, center_y+20), (150, 100, 75), 3)
        
        # Mouth - smiling
        cv2.ellipse(img, (center_x, center_y+35), (25, 12), 0, 0, 180, (100, 50, 50), 3)
        
        # Add some natural skin texture
        texture = np.random.normal(0, 3, (224, 224, 3)).astype(np.uint8)
        img = cv2.add(img, texture)
        
        cv2.imwrite(f'data/real/real_{i:03d}.jpg', img)
        if i < 3:  # Save first few as test images
            cv2.imwrite(f'test_real_{i}.jpg', img)
    
    # Create "fake" faces (with detectable artifacts)
    for i in range(num_samples):
        # Base with different color range
        img = np.random.randint(100, 170, (224, 224, 3), dtype=np.uint8)
        
        # Draw a more artificial face shape
        cv2.ellipse(img, (112, 112), (75, 95), 0, 0, 360, (220, 200, 180), -1)
        
        # Artificial-looking eyes (square-ish)
        cv2.rectangle(img, (85, 95), (105, 110), (50, 50, 50), -1)
        cv2.rectangle(img, (119, 95), (139, 110), (50, 50, 50), -1)
        
        # Blocky pupils
        cv2.rectangle(img, (90, 100), (100, 105), (0, 0, 0), -1)
        cv2.rectangle(img, (124, 100), (134, 105), (0, 0, 0), -1)
        
        # Straight line mouth (unnatural)
        cv2.line(img, (90, 140), (134, 140), (100, 50, 50), 4)
        
        # Add deepfake-like artifacts
        # 1. Blurring around edges
        mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.ellipse(mask, (112, 112), (75, 95), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = mask / 255.0
        
        # 2. Color inconsistencies
        color_shift = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + color_shift, 0, 255).astype(np.uint8)
        
        # 3. Add compression artifacts (blockiness)
        for x in range(0, 224, 16):
            for y in range(0, 224, 16):
                if np.random.random() > 0.7:
                    block_color = np.random.randint(0, 50, 3)
                    cv2.rectangle(img, (x, y), (x+8, y+8), block_color.tolist(), -1)
        
        # 4. Add high-frequency noise (common in GANs)
        noise = np.random.randint(-15, 15, (224, 224, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(f'data/fake/fake_{i:03d}.jpg', img)
        if i < 3:  # Save first few as test images
            cv2.imwrite(f'test_fake_{i}.jpg', img)
    
    print(f"✅ Improved dataset created with {num_samples} real and {num_samples} fake images!")
    print("📁 Real images saved in: data/real/")
    print("📁 Fake images saved in: data/fake/")
    print("📸 Test images saved as: test_real_0.jpg, test_fake_0.jpg, etc.")

def create_simple_faces_for_detection():
    """Create very simple faces that will definitely be detected"""
    create_directories()
    
    for i in range(50):
        # Real - simple clear face
        img = np.full((224, 224, 3), 200, dtype=np.uint8)
        cv2.circle(img, (112, 112), 50, (255, 255, 255), -1)
        cv2.circle(img, (90, 100), 8, (0, 0, 0), -1)
        cv2.circle(img, (134, 100), 8, (0, 0, 0), -1)
        cv2.ellipse(img, (112, 130), (20, 10), 0, 0, 180, (0, 0, 0), 3)
        cv2.imwrite(f'data/real/simple_real_{i:03d}.jpg', img)
        
        # Fake - different shape but still face-like
        img = np.full((224, 224, 3), 150, dtype=np.uint8)
        cv2.rectangle(img, (62, 62), (162, 162), (255, 255, 255), -1)
        cv2.rectangle(img, (80, 90), (100, 110), (0, 0, 0), -1)
        cv2.rectangle(img, (124, 90), (144, 110), (0, 0, 0), -1)
        cv2.line(img, (90, 140), (134, 140), (0, 0, 0), 3)
        # Add noise to make it look fake
        noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)
        cv2.imwrite(f'data/fake/simple_fake_{i:03d}.jpg', img)
    
    print("✅ Simple faces created that will definitely be detected!")

if __name__ == "__main__":
    print("Choose dataset type:")
    print("1. Improved realistic dataset (recommended)")
    print("2. Simple detectable faces (if having face detection issues)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        create_simple_faces_for_detection()
    else:
        create_sample_dataset(200)
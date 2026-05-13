import cv2
import numpy as np
import os

def create_gui_test_images():
    """Create high-quality test images that will work with face detection"""
    print("Creating GUI test images...")
    
    # Create a clear real face
    real_face = np.full((400, 400, 3), 220, dtype=np.uint8)
    
    # Draw a very clear face
    cv2.ellipse(real_face, (200, 200), (120, 150), 0, 0, 360, (255, 230, 200), -1)  # Face
    cv2.ellipse(real_face, (160, 170), (25, 15), 0, 0, 360, (100, 100, 100), -1)    # Left eye
    cv2.ellipse(real_face, (240, 170), (25, 15), 0, 0, 360, (100, 100, 100), -1)    # Right eye
    cv2.circle(real_face, (160, 170), 8, (0, 0, 0), -1)                             # Left pupil
    cv2.circle(real_face, (240, 170), 8, (0, 0, 0), -1)                             # Right pupil
    cv2.ellipse(real_face, (200, 240), (40, 20), 0, 0, 180, (100, 50, 50), 6)       # Mouth
    cv2.line(real_face, (200, 200), (200, 220), (150, 100, 75), 3)                  # Nose
    
    cv2.imwrite('gui_real.jpg', real_face)
    print("✅ Created gui_real.jpg")
    
    # Create a clear fake face
    fake_face = np.full((400, 400, 3), 180, dtype=np.uint8)
    
    # Draw a slightly artificial face
    cv2.rectangle(fake_face, (80, 80), (320, 320), (240, 220, 190), -1)  # Square face
    cv2.rectangle(fake_face, (140, 150), (180, 190), (0, 0, 0), -1)      # Left eye
    cv2.rectangle(fake_face, (220, 150), (260, 190), (0, 0, 0), -1)      # Right eye
    cv2.line(fake_face, (150, 260), (250, 260), (0, 0, 0), 8)            # Mouth
    
    # Add clear artifacts
    noise = np.random.randint(0, 60, (400, 400, 3), dtype=np.uint8)
    fake_face = cv2.addWeighted(fake_face, 0.7, noise, 0.3, 0)
    
    # Add blurring artifacts
    fake_face = cv2.GaussianBlur(fake_face, (9, 9), 0)
    
    cv2.imwrite('gui_fake.jpg', fake_face)
    print("✅ Created gui_fake.jpg")
    
    # Create sample images for the dataset folder too
    os.makedirs('test_samples', exist_ok=True)
    cv2.imwrite('test_samples/sample_real.jpg', real_face)
    cv2.imwrite('test_samples/sample_fake.jpg', fake_face)
    
    print("✅ All test images created successfully!")
    print("📁 Images saved as: gui_real.jpg, gui_fake.jpg")
    print("📁 Sample images saved in: test_samples/")

if __name__ == "__main__":
    create_gui_test_images()
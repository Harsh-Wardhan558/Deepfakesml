import cv2
import numpy as np
import os
from predict import DeepFakePredictor

def test_with_custom_images():
    """Test with manually created images"""
    predictor = DeepFakePredictor()
    
    # Create a simple test image that will definitely have a detectable face
    def create_test_face(is_real=True, save_path="test_custom.jpg"):
        img = np.full((300, 300, 3), 200 if is_real else 150, dtype=np.uint8)
        
        # Always create a clear face that will be detected
        cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)  # Face
        cv2.circle(img, (120, 130), 15, (0, 0, 0), -1)        # Left eye
        cv2.circle(img, (180, 130), 15, (0, 0, 0), -1)        # Right eye
        cv2.ellipse(img, (150, 180), (30, 15), 0, 0, 180, (0, 0, 0), 5)  # Mouth
        
        if not is_real:
            # Add fake artifacts
            noise = np.random.randint(0, 60, (300, 300, 3), dtype=np.uint8)
            img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)
        
        cv2.imwrite(save_path, img)
        return save_path
    
    print("🧪 Testing with guaranteed-detectable faces...")
    
    # Test real face
    real_path = create_test_face(is_real=True, save_path="test_real_detectable.jpg")
    result = predictor.predict_image(real_path)
    print(f"🔍 Real face test:")
    print(f"   Result: {result}")
    
    # Test fake face  
    fake_path = create_test_face(is_real=False, save_path="test_fake_detectable.jpg")
    result = predictor.predict_image(fake_path)
    print(f"🔍 Fake face test:")
    print(f"   Result: {result}")

def visualize_face_detection():
    """Show how face detection works"""
    from utils import extract_face
    
    # Create a test image
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)
    cv2.circle(img, (120, 130), 15, (0, 0, 0), -1)
    cv2.circle(img, (180, 130), 15, (0, 0, 0), -1)
    cv2.ellipse(img, (150, 180), (30, 15), 0, 0, 180, (0, 0, 0), 5)
    
    # Detect face
    face = extract_face(img)
    
    if face is not None:
        print("✅ Face detected successfully!")
        cv2.imwrite("original_face.jpg", img)
        cv2.imwrite("detected_face.jpg", face)
        print("📸 Saved 'original_face.jpg' and 'detected_face.jpg'")
    else:
        print("❌ No face detected - this explains the issue!")
        
        # Try with a more obvious face
        img_obvious = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.circle(img_obvious, (150, 150), 70, (0, 0, 0), -1)  # Black face on white background
        cv2.circle(img_obvious, (130, 130), 10, (255, 255, 255), -1)  # White eyes
        cv2.circle(img_obvious, (170, 130), 10, (255, 255, 255), -1)
        
        face_obvious = extract_face(img_obvious)
        if face_obvious is not None:
            print("✅ Obvious face detected!")
            cv2.imwrite("obvious_face.jpg", img_obvious)
        else:
            print("❌ Even obvious face not detected - check Haar cascade file")

if __name__ == "__main__":
    print("🔧 Testing Improved Deepfake Detection")
    print("=" * 50)
    
    # First test face detection
    print("\n1. Testing face detection...")
    visualize_face_detection()
    
    # Then test predictions
    print("\n2. Testing predictions...")
    test_with_custom_images()
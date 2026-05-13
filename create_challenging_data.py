import os
import cv2
import numpy as np
from utils import create_directories

def create_challenging_dataset(num_samples=300):
    """Create a more challenging and realistic dataset"""
    create_directories()
    
    print("Creating challenging dataset with real-world variations...")
    
    # Clear existing data first
    for folder in ['data/real', 'data/fake']:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    os.remove(os.path.join(folder, file))
    
    # Create more realistic and varied faces
    for i in range(num_samples):
        # Real faces with natural variations
        create_real_face(i)
        
        # Fake faces with subtle artifacts (like real deepfakes)
        create_fake_face(i)
    
    print(f"✅ Challenging dataset created with {num_samples} real and fake images!")
    print("📊 This dataset will be harder to classify, making the model more robust.")

def create_real_face(index):
    """Create realistic face with natural variations"""
    try:
        # Random background and skin tones
        bg_color = np.random.randint(100, 200, 3)
        skin_tone = np.random.randint(150, 220, 3)
        
        img = np.full((224, 224, 3), bg_color, dtype=np.uint8)
        
        # Face position variations
        center_x = 112 + np.random.randint(-10, 10)
        center_y = 112 + np.random.randint(-10, 10)
        face_width = np.random.randint(70, 90)
        face_height = np.random.randint(90, 110)
        
        # Natural face oval
        cv2.ellipse(img, (center_x, center_y), (face_width, face_height), 0, 0, 360, skin_tone.tolist(), -1)
        
        # Eye variations
        eye_y_offset = np.random.randint(-5, 5)
        left_eye = (center_x-35, center_y-15 + eye_y_offset)
        right_eye = (center_x+35, center_y-15 + eye_y_offset)
        
        # Natural eyes
        cv2.ellipse(img, left_eye, (12, 8), 0, 0, 360, (80, 80, 80), -1)
        cv2.ellipse(img, right_eye, (12, 8), 0, 0, 360, (80, 80, 80), -1)
        
        # Pupils
        cv2.circle(img, left_eye, 4, (0, 0, 0), -1)
        cv2.circle(img, right_eye, 4, (0, 0, 0), -1)
        
        # Mouth variations
        mouth_shape = np.random.choice(['smile', 'neutral', 'slight_smile'])
        mouth_y = center_y + 35 + np.random.randint(-5, 5)
        
        if mouth_shape == 'smile':
            cv2.ellipse(img, (center_x, mouth_y), (25, 12), 0, 0, 180, (100, 50, 50), 3)
        elif mouth_shape == 'neutral':
            cv2.line(img, (center_x-20, mouth_y), (center_x+20, mouth_y), (100, 50, 50), 3)
        else:  # slight_smile
            cv2.ellipse(img, (center_x, mouth_y), (20, 8), 0, 0, 180, (100, 50, 50), 3)
        
        # Add natural skin texture
        texture = np.random.normal(0, 5, (224, 224, 3)).astype(np.uint8)
        img = cv2.add(img, texture)
        
        # Random lighting variations
        brightness = np.random.uniform(0.9, 1.1)
        img = np.clip(img.astype(np.float64) * brightness, 0, 255).astype(np.uint8)
        
        cv2.imwrite(f'data/real/real_{index:03d}.jpg', img)
        
    except Exception as e:
        print(f"Error creating real face {index}: {e}")

def create_fake_face(index):
    """Create fake face with subtle deepfake-like artifacts"""
    try:
        # Start with a realistic base (similar to real faces)
        bg_color = np.random.randint(100, 200, 3)
        skin_tone = np.random.randint(150, 220, 3)
        
        img = np.full((224, 224, 3), bg_color, dtype=np.uint8)
        
        center_x, center_y = 112, 112
        
        # Face shape (slightly different from real)
        cv2.ellipse(img, (center_x, center_y), (75, 95), 0, 0, 360, skin_tone.tolist(), -1)
        
        # Eyes (slightly artificial)
        cv2.ellipse(img, (center_x-35, center_y-15), (14, 9), 0, 0, 360, (70, 70, 70), -1)
        cv2.ellipse(img, (center_x+35, center_y-15), (14, 9), 0, 0, 360, (70, 70, 70), -1)
        
        # Pupils
        cv2.circle(img, (center_x-35, center_y-15), 5, (0, 0, 0), -1)
        cv2.circle(img, (center_x+35, center_y-15), 5, (0, 0, 0), -1)
        
        # Mouth (slightly unnatural)
        cv2.ellipse(img, (center_x, center_y+35), (22, 10), 0, 0, 180, (90, 45, 45), 3)
        
        # ===== DEEPFAKE ARTIFACTS =====
        
        # 1. Subtle color mismatches (common in deepfakes)
        if np.random.random() > 0.3:
            # Convert to float for safe operations
            img_float = img.astype(np.float64)
            color_shift = np.random.randint(-15, 15, (224, 224, 3))
            img_float = np.clip(img_float + color_shift, 0, 255)
            img = img_float.astype(np.uint8)
        
        # 2. Blurring artifacts around face edges
        if np.random.random() > 0.4:
            # Create blur mask around face edges
            mask = np.zeros((224, 224), dtype=np.uint8)
            cv2.ellipse(mask, (center_x, center_y), (78, 98), 0, 0, 360, 255, -1)
            cv2.ellipse(mask, (center_x, center_y), (72, 92), 0, 0, 360, 0, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            mask = mask / 255.0
            
            # Apply blur to edge regions
            blurred = cv2.GaussianBlur(img, (15, 15), 0)
            for c in range(3):
                img[:,:,c] = (1 - mask) * img[:,:,c] + mask * blurred[:,:,c]
        
        # 3. High-frequency noise (GAN artifacts)
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 8, (224, 224, 3)).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # 4. Slight misalignment (temporal artifacts simulated)
        if np.random.random() > 0.7:
            # Shift features slightly
            dx, dy = np.random.randint(-3, 3, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M, (224, 224))
        
        # 5. Compression artifacts (blockiness)
        if np.random.random() > 0.6:
            quality = np.random.randint(30, 70)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            if result:
                img = cv2.imdecode(encimg, 1)
        
        cv2.imwrite(f'data/fake/fake_{index:03d}.jpg', img)
        
    except Exception as e:
        print(f"Error creating fake face {index}: {e}")

def add_real_world_variations():
    """Add real-world variations to existing dataset"""
    try:
        real_files = [f for f in os.listdir('data/real') if f.endswith(('.jpg', '.png', '.jpeg'))]
        fake_files = [f for f in os.listdir('data/fake') if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Found {len(real_files)} real files and {len(fake_files)} fake files")
        print("Adding real-world variations to existing dataset...")
        
        # Process real images
        for i, file in enumerate(real_files[:100]):  # Modify first 100 real images
            img_path = os.path.join('data/real', file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Random transformations
            if np.random.random() > 0.5:
                # Color variations using safe operations
                img_float = img.astype(np.float64)
                # Adjust brightness and contrast
                alpha = np.random.uniform(0.8, 1.2)  # Contrast
                beta = np.random.randint(-20, 20)    # Brightness
                img_float = np.clip(alpha * img_float + beta, 0, 255)
                img = img_float.astype(np.uint8)
            
            if np.random.random() > 0.3:
                # Safe noise addition
                noise = np.random.normal(0, 5, img.shape).astype(np.float64)
                img_float = img.astype(np.float64)
                img_float = np.clip(img_float + noise, 0, 255)
                img = img_float.astype(np.uint8)
            
            cv2.imwrite(img_path, img)
            
            if i % 20 == 0:
                print(f"Processed {i} real images...")
        
        # Process fake images
        for i, file in enumerate(fake_files[:100]):  # Modify first 100 fake images
            img_path = os.path.join('data/fake', file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Add additional artifacts to fake images
            if np.random.random() > 0.4:
                # Add more compression artifacts
                quality = np.random.randint(40, 80)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                result, encimg = cv2.imencode('.jpg', img, encode_param)
                if result:
                    img = cv2.imdecode(encimg, 1)
            
            cv2.imwrite(img_path, img)
            
            if i % 20 == 0:
                print(f"Processed {i} fake images...")
        
        print("✅ Added real-world variations to dataset")
        
    except Exception as e:
        print(f"Error in add_real_world_variations: {e}")
        import traceback
        traceback.print_exc()

def verify_dataset():
    """Verify the created dataset"""
    print("\n🔍 Verifying dataset...")
    
    real_files = [f for f in os.listdir('data/real') if f.endswith(('.jpg', '.png', '.jpeg'))]
    fake_files = [f for f in os.listdir('data/fake') if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Real images: {len(real_files)}")
    print(f"Fake images: {len(fake_files)}")
    
    # Check if images are readable
    valid_real = 0
    valid_fake = 0
    
    for file in real_files[:5]:  # Check first 5
        img = cv2.imread(os.path.join('data/real', file))
        if img is not None:
            valid_real += 1
    
    for file in fake_files[:5]:
        img = cv2.imread(os.path.join('data/fake', file))
        if img is not None:
            valid_fake += 1
    
    print(f"Readable real images: {valid_real}/5")
    print(f"Readable fake images: {valid_fake}/5")
    
    if valid_real > 0 and valid_fake > 0:
        print("✅ Dataset verification passed!")
        return True
    else:
        print("❌ Dataset verification failed!")
        return False

if __name__ == "__main__":
    try:
        create_challenging_dataset(200)  # Reduced to 200 for faster testing
        add_real_world_variations()
        verify_dataset()
        
        print("\n🎉 Dataset creation completed successfully!")
        print("📁 You can now train your model with:")
        print("   python train_advanced.py")
        
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()
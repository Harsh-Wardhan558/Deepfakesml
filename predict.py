import torch
import cv2
import numpy as np
from model import create_model
from utils import extract_face_robust  # Changed to robust version
import torchvision.transforms as transforms
import os

class DeepFakePredictor:
    def __init__(self, model_path='models/deepfake_model_advanced.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Model loaded from {model_path}")
        else:
            # Try to find any trained model
            model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
            if model_files:
                model_path = os.path.join('models', model_files[0])
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Model loaded from {model_path}")
            else:
                print("❌ No model file found. Please train the model first.")
                return
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path):
        """Predict if a single image is real or fake"""
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not read image"}
            
            # Extract face using robust method
            face = extract_face_robust(image)
            if face is None:
                # If no face detected, use the entire image
                print("⚠️ No face detected, using entire image")
                face = cv2.resize(image, (224, 224))
            
            return self.predict_frame(face)
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_frame(self, frame):
        """Predict for a single frame (numpy array)"""
        try:
            # Ensure the frame is the right size and format
            if frame.shape[:2] != (224, 224):
                frame = cv2.resize(frame, (224, 224))
            
            face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                fake_prob = probabilities[0][1].item()
                real_prob = probabilities[0][0].item()
            
            result = "FAKE" if fake_prob > 0.5 else "REAL"
            confidence = max(fake_prob, real_prob)
            
            return {
                'prediction': result,
                'confidence': round(confidence, 4),
                'fake_probability': round(fake_prob, 4),
                'real_probability': round(real_prob, 4),
                'used_face_detection': 'yes' if fake_prob != 0.5 else 'no'
            }
        except Exception as e:
            return {"error": f"Frame prediction failed: {str(e)}"}

def test_prediction():
    """Test the predictor with sample data"""
    predictor = DeepFakePredictor()
    
    # Create a simple test image that will definitely work
    def create_simple_test_image(filename, is_real=True):
        img = np.full((300, 300, 3), 200 if is_real else 150, dtype=np.uint8)
        
        # Create a very clear face
        cv2.circle(img, (150, 150), 80, (255, 220, 180), -1)  # Face
        cv2.circle(img, (120, 130), 15, (0, 0, 0), -1)        # Left eye
        cv2.circle(img, (180, 130), 15, (0, 0, 0), -1)        # Right eye
        cv2.ellipse(img, (150, 180), (25, 12), 0, 0, 180, (0, 0, 0), 4)  # Mouth
        
        if not is_real:
            # Add noise to make it look fake
            noise = np.random.randint(0, 50, (300, 300, 3), dtype=np.uint8)
            img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)
        
        cv2.imwrite(filename, img)
        return filename
    
    # Test with guaranteed images
    real_path = create_simple_test_image('test_real_clear.jpg', True)
    fake_path = create_simple_test_image('test_fake_clear.jpg', False)
    
    print("🧪 Testing with clear face images:")
    
    result = predictor.predict_image(real_path)
    print(f"   Real image: {result}")
    
    result = predictor.predict_image(fake_path)
    print(f"   Fake image: {result}")

if __name__ == "__main__":
    test_prediction()
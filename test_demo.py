import os
import sys
from create_sample_data import create_sample_dataset
from train import train_model
from predict import DeepFakePredictor

def run_complete_demo():
    """Run a complete demo of the deepfake detection system"""
    print("=" * 50)
    print("🤖 DEEPFAKE DETECTION SYSTEM - COMPLETE DEMO")
    print("=" * 50)
    
    # Step 1: Create sample data
    print("\n1️⃣ Creating sample dataset...")
    create_sample_dataset(100)
    
    # Step 2: Train the model
    print("\n2️⃣ Training the model...")
    train_model("data/real", "data/fake")
    
    # Step 3: Test the model
    print("\n3️⃣ Testing the model...")
    predictor = DeepFakePredictor()
    
    # Test on a few samples
    print("\n📊 Testing predictions:")
    test_files = [
        ('data/real/real_001.jpg', 'Real'),
        ('data/fake/fake_001.jpg', 'Fake'),
        ('data/real/real_010.jpg', 'Real'),
        ('data/fake/fake_010.jpg', 'Fake')
    ]
    
    for file_path, expected in test_files:
        if os.path.exists(file_path):
            result = predictor.predict_image(file_path)
            if 'error' not in result:
                status = "✅" if result['prediction'] == expected.upper() else "❌"
                print(f"   {status} {file_path}:")
                print(f"      Prediction: {result['prediction']} (Expected: {expected})")
                print(f"      Confidence: {result['confidence']:.2%}")
    
    print("\n🎉 Demo completed! Check 'models/' for trained model and 'results/' for training graphs.")

if __name__ == "__main__":
    run_complete_demo()
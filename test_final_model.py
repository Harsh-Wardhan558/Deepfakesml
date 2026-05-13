import cv2
import numpy as np
import os
from predict import DeepFakePredictor
import matplotlib.pyplot as plt

def comprehensive_test():
    """Comprehensive test of the trained model"""
    print("🧪 COMPREHENSIVE MODEL TEST")
    print("=" * 50)
    
    # Initialize predictor
    predictor = DeepFakePredictor('models/deepfake_model_advanced.pth')
    
    # Test on multiple samples
    real_samples = []
    fake_samples = []
    
    # Test real images
    print("\n🔍 Testing REAL images:")
    real_correct = 0
    real_total = 0
    
    real_files = [f for f in os.listdir('data/real') if f.endswith(('.jpg', '.png', '.jpeg'))][:20]
    
    for file in real_files:
        result = predictor.predict_image(os.path.join('data/real', file))
        if 'error' not in result:
            real_total += 1
            if result['prediction'] == 'REAL':
                real_correct += 1
                real_samples.append(result['fake_probability'])
            else:
                print(f"   ❌ Misclassified: {file} as FAKE (prob: {result['fake_probability']:.3f})")
    
    real_accuracy = real_correct / real_total if real_total > 0 else 0
    
    # Test fake images
    print("\n🔍 Testing FAKE images:")
    fake_correct = 0
    fake_total = 0
    
    fake_files = [f for f in os.listdir('data/fake') if f.endswith(('.jpg', '.png', '.jpeg'))][:20]
    
    for file in fake_files:
        result = predictor.predict_image(os.path.join('data/fake', file))
        if 'error' not in result:
            fake_total += 1
            if result['prediction'] == 'FAKE':
                fake_correct += 1
                fake_samples.append(result['fake_probability'])
            else:
                print(f"   ❌ Misclassified: {file} as REAL (prob: {result['fake_probability']:.3f})")
    
    fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
    
    # Print results
    print("\n📊 TEST RESULTS:")
    print("=" * 30)
    print(f"Real Images: {real_correct}/{real_total} correct ({real_accuracy*100:.1f}%)")
    print(f"Fake Images: {fake_correct}/{fake_total} correct ({fake_accuracy*100:.1f}%)")
    print(f"Overall Accuracy: {(real_correct + fake_correct)/(real_total + fake_total)*100:.1f}%")
    
    # Create visualization
    if real_samples and fake_samples:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(real_samples, alpha=0.7, label='Real Images', bins=15, color='green', edgecolor='black')
        plt.hist(fake_samples, alpha=0.7, label='Fake Images', bins=15, color='red', edgecolor='black')
        plt.xlabel('Fake Probability')
        plt.ylabel('Count')
        plt.title('Probability Distribution\n(Lower = Real, Higher = Fake)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Box plot
        data = [real_samples, fake_samples]
        plt.boxplot(data, labels=['Real', 'Fake'])
        plt.ylabel('Fake Probability')
        plt.title('Probability Distribution Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/test_results_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return real_accuracy, fake_accuracy

def create_demo_images():
    """Create demo images to test the model"""
    print("\n🎨 Creating demo test images...")
    
    # Create a clear real face
    real_face = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.circle(real_face, (150, 150), 80, (255, 220, 180), -1)
    cv2.circle(real_face, (120, 130), 12, (0, 0, 0), -1)
    cv2.circle(real_face, (180, 130), 12, (0, 0, 0), -1)
    cv2.ellipse(real_face, (150, 180), (25, 12), 0, 0, 180, (0, 0, 0), 4)
    cv2.imwrite('demo_real.jpg', real_face)
    
    # Create a clear fake face with artifacts
    fake_face = np.full((300, 300, 3), 160, dtype=np.uint8)
    cv2.rectangle(fake_face, (70, 70), (230, 230), (220, 200, 170), -1)
    cv2.rectangle(fake_face, (110, 120), (130, 140), (0, 0, 0), -1)
    cv2.rectangle(fake_face, (170, 120), (190, 140), (0, 0, 0), -1)
    cv2.line(fake_face, (120, 190), (180, 190), (0, 0, 0), 5)
    
    # Add artifacts
    noise = np.random.randint(0, 40, (300, 300, 3), dtype=np.uint8)
    fake_face = cv2.add(fake_face, noise)
    fake_face = cv2.GaussianBlur(fake_face, (7, 7), 0)
    
    cv2.imwrite('demo_fake.jpg', fake_face)
    
    print("✅ Demo images created: 'demo_real.jpg' and 'demo_fake.jpg'")

def test_demo_images():
    """Test the model on demo images"""
    print("\n🧪 Testing on demo images...")
    
    predictor = DeepFakePredictor('models/deepfake_model_advanced.pth')
    
    if os.path.exists('demo_real.jpg'):
        result = predictor.predict_image('demo_real.jpg')
        print(f"📷 Demo Real Image: {result}")
    
    if os.path.exists('demo_fake.jpg'):
        result = predictor.predict_image('demo_fake.jpg')
        print(f"📷 Demo Fake Image: {result}")

def performance_analysis():
    """Analyze model performance characteristics"""
    print("\n📈 PERFORMANCE ANALYSIS")
    print("=" * 30)
    
    predictor = DeepFakePredictor('models/deepfake_model_advanced.pth')
    
    # Test confidence levels
    confidences = []
    predictions = []
    
    test_files = []
    test_files.extend([os.path.join('data/real', f) for f in os.listdir('data/real')[:10]])
    test_files.extend([os.path.join('data/fake', f) for f in os.listdir('data/fake')[:10]])
    
    for file in test_files:
        result = predictor.predict_image(file)
        if 'error' not in result:
            confidences.append(result['confidence'])
            predictions.append(1 if result['prediction'] == 'FAKE' else 0)
    
    if confidences:
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        print(f"Average Confidence: {avg_confidence:.3f} (+/- {confidence_std:.3f})")
        print(f"Confidence Range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"High Confidence (>0.9): {sum(c > 0.9 for c in confidences)}/{len(confidences)}")
        print(f"Low Confidence (<0.7): {sum(c < 0.7 for c in confidences)}/{len(confidences)}")
        
        # Confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(avg_confidence, color='red', linestyle='--', label=f'Average: {avg_confidence:.3f}')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Model Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    print("🚀 FINAL MODEL EVALUATION")
    print("This will test your improved deepfake detection model!")
    
    # Create demo images
    create_demo_images()
    
    # Run comprehensive tests
    real_acc, fake_acc = comprehensive_test()
    
    # Test demo images
    test_demo_images()
    
    # Performance analysis
    performance_analysis()
    
    print("\n🎉 EVALUATION COMPLETED!")
    print("📁 Check the 'results/' folder for analysis graphs")
    print("💡 Your model is now ready for real deepfake detection tasks!")
    
    # Final assessment
    overall_acc = (real_acc + fake_acc) / 2
    if overall_acc > 0.85:
        print("🏆 EXCELLENT! Your model performs very well!")
    elif overall_acc > 0.75:
        print("✅ GOOD! Your model performs well!")
    else:
        print("📚 DECENT! Consider training with more data or epochs.")
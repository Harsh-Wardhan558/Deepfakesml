import torch
import numpy as np
from model import create_model
from data_loader import get_data_loaders
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_model(real_dir, fake_dir, model_path='models/deepfake_model.pth'):
    """Comprehensive model evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get data loaders
    _, test_loader = get_data_loaders(real_dir, fake_dir, batch_size=16)
    
    # Evaluation
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels)
    
    print("📊 MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total Test Samples: {len(all_labels)}")
    print(f"Real Samples: {np.sum(all_labels == 0)}")
    print(f"Fake Samples: {np.sum(all_labels == 1)}")
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['Real', 'Fake']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Probability distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    real_probs = all_probabilities[all_labels == 0, 1]  # Fake probabilities for real images
    fake_probs = all_probabilities[all_labels == 1, 1]  # Fake probabilities for fake images
    
    plt.hist(real_probs, alpha=0.7, label='Real Images', bins=20, color='blue')
    plt.hist(fake_probs, alpha=0.7, label='Fake Images', bins=20, color='red')
    plt.xlabel('Fake Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # ROC curve (simplified)
    thresholds = np.linspace(0, 1, 50)
    tpr = []
    fpr = []
    
    for thresh in thresholds:
        preds = (all_probabilities[:, 1] > thresh).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/probability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate additional metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    print(f"\n🎯 Additional Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return accuracy, all_probabilities

if __name__ == "__main__":
    real_dir = "data/real"
    fake_dir = "data/fake"
    
    evaluate_model(real_dir, fake_dir)
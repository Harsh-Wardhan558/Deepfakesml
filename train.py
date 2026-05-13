import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from model import create_model, count_parameters
from data_loader import get_data_loaders

def train_model(real_dir, fake_dir, model_save_path='models/deepfake_model.pth'):
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create data loaders with smaller batch size for better generalization
        train_loader, val_loader = get_data_loaders(real_dir, fake_dir, batch_size=8)  # Reduced from 16 to 8
        
        # Create model
        model = create_model(device)
        
        # Print model info
        trainable_params = count_parameters(model)
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Loss function and optimizer with regularization
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Added weight decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        
        # Training variables
        train_losses = []
        val_accuracies = []
        best_accuracy = 0.0
        
        # Early stopping
        patience = 5
        patience_counter = 0
        
        # Training loop - increased epochs
        num_epochs = 15
        
        print("Starting training...")
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(device), labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                running_loss += loss.item()
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*correct_train/total_train:.1f}%'
                })
            
            train_accuracy = 100 * correct_train / total_train
            avg_loss = running_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            correct_val = 0
            total_val = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct_val / total_val
            avg_val_loss = val_loss / len(val_loader)
            epoch_time = time.time() - start_time
            
            train_losses.append(avg_loss)
            val_accuracies.append(val_accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Time: {epoch_time:.1f}s')
            print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print('-' * 50)
            
            # Update learning rate
            scheduler.step(val_accuracy)
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f'💾 New best model saved with validation accuracy: {val_accuracy:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
        plt.title('Validation Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 3, 3)
        # Add training accuracy plot
        train_accuracies = [min(acc, 100) for acc in [train_accuracy] * len(train_losses)]  # Simplified
        plt.plot(train_accuracies, 'r-', label='Training Accuracy', linewidth=2)
        plt.plot(val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/training_results_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"🎉 Training completed! Best validation accuracy: {best_accuracy:.2f}%")
        
        # Final model evaluation
        print("\n📊 Final Model Evaluation:")
        print(f"   - Best Validation Accuracy: {best_accuracy:.2f}%")
        print(f"   - Total Epochs Trained: {epoch+1}")
        print(f"   - Model saved as: {model_save_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure you have created sample data first!")

if __name__ == "__main__":
    real_dir = "data/real"
    fake_dir = "data/fake"
    
    print("🚀 Starting Improved Deepfake Detection Training")
    print(f"Real data directory: {real_dir}")
    print(f"Fake data directory: {fake_dir}")
    
    train_model(real_dir, fake_dir)
    # In model.py - Update the model for better regularization
class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepFakeDetector, self).__init__()
        
        # Use a pre-trained ResNet18 model
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze more layers to prevent overfitting
        for param in list(self.backbone.parameters())[:-30]:  # Freeze all but last 30 layers
            param.requires_grad = False
        
        # Replace the final layer with more regularization
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),  # Added batch norm
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import numpy as np
from model import create_model, count_parameters
from data_loader import get_data_loaders

def train_advanced_model(real_dir, fake_dir, model_save_path='models/deepfake_model_advanced.pth'):
    """Advanced training with better regularization"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Smaller batch size for better generalization
        train_loader, val_loader = get_data_loaders(real_dir, fake_dir, batch_size=8)
        
        # Create model
        model = create_model(device)
        
        print(f"Trainable parameters: {count_parameters(model):,}")
        
        # Advanced optimizer with more regularization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # AdamW with decoupled weight decay
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        # Training variables
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_accuracy = 0.0
        
        # Early stopping
        patience = 8
        patience_counter = 0
        
        num_epochs = 20
        
        print("🚀 Starting Advanced Training...")
        
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
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Calculate metrics
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                running_loss += loss.item()
                current_acc = 100 * correct_train / total_train
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.1f}%'
                })
            
            train_accuracy = 100 * correct_train / total_train
            avg_train_loss = running_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
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
            
            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'  Time: {epoch_time:.1f}s, LR: {current_lr:.6f}')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print('-' * 60)
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f'💾 Best model saved! Val Accuracy: {val_accuracy:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Plot comprehensive results
        plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies)
        
        print(f"\n🎉 Training completed!")
        print(f"   Best Validation Accuracy: {best_accuracy:.2f}%")
        print(f"   Final Training Accuracy: {train_accuracy:.2f}%")
        print(f"   Model saved as: {model_save_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot comprehensive training results"""
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Combined plot
    plt.subplot(1, 3, 3)
    # Normalize losses for combined plot
    norm_train_loss = [loss/max(train_losses) for loss in train_losses]
    plt.plot(norm_train_loss, 'b--', alpha=0.7, label='Norm. Train Loss')
    plt.plot([acc/100 for acc in train_accuracies], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot([acc/100 for acc in val_accuracies], 'r-', label='Val Accuracy', linewidth=2)
    plt.title('Combined Metrics (Normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/advanced_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    real_dir = "data/real"
    fake_dir = "data/fake"
    
    print("🚀 Starting Advanced Deepfake Detection Training")
    train_advanced_model(real_dir, fake_dir)
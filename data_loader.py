import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

class DeepFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, max_samples_per_class=500):
        """
        real_dir: directory containing real face images
        fake_dir: directory containing fake face images
        """
        self.transform = transform
        self.samples = []
        
        # Load real images
        if os.path.exists(real_dir):
            real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))]
            real_files = real_files[:max_samples_per_class]
            self.samples.extend([(f, 0) for f in real_files])  # 0 for real
            print(f"Loaded {len(real_files)} real images")
        else:
            print(f"Warning: Real directory {real_dir} not found")
        
        # Load fake images
        if os.path.exists(fake_dir):
            fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))]
            fake_files = fake_files[:max_samples_per_class]
            self.samples.extend([(f, 1) for f in fake_files])  # 1 for fake
            print(f"Loaded {len(fake_files)} fake images")
        else:
            print(f"Warning: Fake directory {fake_dir} not found")
        
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                # If image can't be read, return a black image
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform:
                image = self.transform(image)
            return image, label

def get_data_loaders(real_dir, fake_dir, batch_size=16):
    """
    Create data loaders for training and validation
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = DeepFakeDataset(real_dir, fake_dir)
    
    if len(full_dataset) == 0:
        raise ValueError("No data found! Please check your data directories.")
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
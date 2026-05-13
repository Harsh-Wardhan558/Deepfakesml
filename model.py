import torch
import torch.nn as nn
import torchvision.models as models

class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepFakeDetector, self).__init__()
        
        # Use a pre-trained ResNet18 model
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers to prevent overfitting
        layers_to_freeze = len(list(self.backbone.parameters())) - 25
        for i, param in enumerate(self.backbone.parameters()):
            if i < layers_to_freeze:
                param.requires_grad = False
        
        # Replace the final layer with more regularization
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),  # Dropout for regularization
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_model(device='cpu'):
    model = DeepFakeDetector()
    return model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
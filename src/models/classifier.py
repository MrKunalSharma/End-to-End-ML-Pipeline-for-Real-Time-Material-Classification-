import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.helpers import load_config

config = load_config()

class MaterialClassifier(nn.Module):
    def __init__(self, num_classes=None, architecture='resnet18', pretrained=True):
        super(MaterialClassifier, self).__init__()
        
        if num_classes is None:
            num_classes = config['model']['num_classes']
        
        self.architecture = architecture
        
        # Load pretrained model
        if architecture == 'resnet18':
            if pretrained:
                self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                self.base_model = resnet18(weights=None)
            in_features = self.base_model.fc.in_features
            
            # Replace final layer
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            
        elif architecture == 'mobilenet_v2':
            if pretrained:
                self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            else:
                self.base_model = mobilenet_v2(weights=None)
            in_features = self.base_model.classifier[1].in_features
            
            # Replace classifier
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            
        elif architecture == 'efficientnet_b0':
            # Note: You might need to install timm: pip install timm
            try:
                import timm
                self.base_model = timm.create_model('efficientnet_b0', pretrained=pretrained)
                in_features = self.base_model.classifier.in_features
                
                self.base_model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_classes)
                )
            except ImportError:
                print("timm not installed. Using ResNet18 instead.")
                return self.__init__(num_classes, 'resnet18', pretrained)
        
    def forward(self, x):
        return self.base_model(x)
    
    def get_features(self, x):
        """Extract features before the final classification layer"""
        if self.architecture == 'resnet18':
            # Forward through all layers except the final fc
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)
            
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
            
            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)
            
            return x
        else:
            # For other architectures, return the input to final layer
            # This is a simplified version
            return x

def create_model(architecture=None, num_classes=None, pretrained=True):
    """Factory function to create model"""
    if architecture is None:
        architecture = config['model']['architecture']
    
    model = MaterialClassifier(
        num_classes=num_classes,
        architecture=architecture,
        pretrained=pretrained
    )
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

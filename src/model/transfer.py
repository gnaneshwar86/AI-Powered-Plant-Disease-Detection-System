import os
import torch
import torch.nn as nn
import torchvision.models as models
from src.model.cnn import CNN

class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes=39, pretrained=True):
        super(ResNet50Transfer, self).__init__()
        # weights argument is standard in torchvision>=0.13
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class EfficientNetB0Transfer(nn.Module):
    def __init__(self, num_classes=39, pretrained=True):
        super(EfficientNetB0Transfer, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_model(model_name="cnn", num_classes=39, weights_path=None, device="cpu"):
    """
    Factory function to load and initialize models.
    Supports 'cnn', 'resnet50', and 'efficientnet_b0'.
    If weights_path is provided and exists, loads the state dict.
    Otherwise, initializes with default weights and logs a warning.
    """
    model_name = model_name.lower()
    
    if model_name == "cnn":
        model = CNN(num_classes)
    elif model_name == "resnet50":
        model = ResNet50Transfer(num_classes, pretrained=True)
    elif model_name == "efficientnet_b0":
        model = EfficientNetB0Transfer(num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: cnn, resnet50, efficientnet_b0")
    
    if weights_path and os.path.exists(weights_path):
        try:
            # Load state dict
            state_dict = torch.load(weights_path, map_location=device)
            # Support loading state dicts that might be wrapped or unwrapped
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            print(f"Successfully loaded trained weights for {model_name} from {weights_path}")
        except Exception as e:
            print(f"Warning: Failed to load weights from {weights_path} with error: {e}. "
                  f"Using default/pretrained base weights for {model_name}.")
    else:
        print(f"Warning: Weights path '{weights_path}' not found. "
              f"Using default initialized model for {model_name}.")
        
    model = model.to(device)
    model.eval()
    return model

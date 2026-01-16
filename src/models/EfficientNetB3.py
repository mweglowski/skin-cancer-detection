import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class EfficientNetB3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)

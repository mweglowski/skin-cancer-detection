import timm
import torch.nn as nn

class EfficientNetV2S(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = timm.create_model(
            "efficientnetv2_s",
            pretrained=True,
            num_classes=0
        )
        in_features = self.model.num_features

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

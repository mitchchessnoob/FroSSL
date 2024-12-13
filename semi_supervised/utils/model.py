import torch.nn as nn
import torchvision.models as models
class SSLModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load ResNet18 backbone
        self.backbone = models.resnet18(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final FC layer

        # Add new FC layer for classification
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        if return_features:
            return features, self.classifier(features)
        return self.classifier(features)

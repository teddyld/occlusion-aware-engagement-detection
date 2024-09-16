import torch.nn as nn
import torchvision.models as models

class InceptionNetV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT
        self.model = models.inception_v3(weights=weights)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        return x
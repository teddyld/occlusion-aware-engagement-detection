import torch.nn as nn
import torchvision.models as models

class EfficientNetB7(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.EfficientNet_B7_Weights.DEFAULT
        self.model = models.efficientnet_b7(weights=weights)
        n_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(n_features, 7)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        return x
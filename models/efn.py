import torch.nn as nn
import timm

class EfficientNetB0(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b0.in1k', pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 7)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        return x
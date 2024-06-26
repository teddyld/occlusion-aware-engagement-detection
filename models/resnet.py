import torch.nn as nn
import timm

class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('resnet50.a1_in1k', pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 7)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        return x
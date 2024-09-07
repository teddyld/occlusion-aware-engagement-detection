import torch.nn as nn
import torchvision.models as models

class Vgg16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = models.VGG16_BN_Weights.DEFAULT
        self.model = models.vgg16_bn(weights=weights)
        n_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        return x
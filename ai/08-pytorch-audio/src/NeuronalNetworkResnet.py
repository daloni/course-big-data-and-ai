import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

class NeuronalNetwork(nn.Module):
    def __init__(self):
        super(NeuronalNetwork, self).__init__()

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for name, param in resnet.named_parameters():
            param.requires_grad = False

            # Unfreeze the deep layers
            if 'layer4' in name:
                param.requires_grad = True

        for param in resnet.fc.parameters():
            param.requires_grad = True

        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 2048)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            resnet,

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 50)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x
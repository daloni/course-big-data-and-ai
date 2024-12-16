import torch.nn as nn
import torch.nn.functional as F

class NeuronalNetwork(nn.Module):
    def __init__(self):
        super(NeuronalNetwork, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=10, stride=4, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=3),

            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.flatten_size = 384

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.flatten_size)
        x = self.classifier(x)
        return x

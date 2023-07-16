# resnet34

import torchvision
import torch.nn as nn


class ResNet34(nn.Module):
    def __init__(self, input_size=128, embedding_size=64, pretrained=False):
        nn.Module.__init__(self)
        self.model = torchvision.models.resnet34(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(
            in_features=512, out_features=embedding_size, bias=True
        )

    def forward(self, x):
        return self.model(x)

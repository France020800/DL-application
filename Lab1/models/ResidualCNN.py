import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, num_classes, planes, num_blocks=2, block=BasicBlock):
        super().__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.residual_layers = nn.ModuleList()
        self.residual_layers.append(self._make_layer(block, 64, blocks=num_blocks))
        for plane in planes:
            self.residual_layers.append(self._make_layer(block, plane, blocks=num_blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(planes[-1], num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        blocks = 1 if planes >= 512 else blocks
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for residual_layer in self.residual_layers:
            x = residual_layer(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x
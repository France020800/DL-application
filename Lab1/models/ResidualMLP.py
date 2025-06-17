import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.linear2 = nn.Linear(in_features, in_features)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x += identity
        return x


class ResidualMLP(nn.Module):
    def __init__(self, in_features, out_features, width, depth):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, width))
        for _ in range(depth-1):
            self.layers.append(ResidualBlock(width))
        self.layers.append(nn.Linear(width, out_features))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)  # Last layer
        return x

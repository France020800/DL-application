import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers[:-1], layers[1:]):
            self.layers.append(nn.Linear(in_features, out_features))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)  # Last layer
        return x

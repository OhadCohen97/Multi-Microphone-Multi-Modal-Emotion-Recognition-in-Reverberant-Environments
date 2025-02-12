import torch
import torch.nn as nn


class ClassificationLayer(nn.Module):
    def __init__(self, num_feat=768, n_classes=8):
        super(ClassificationLayer, self).__init__()

        self.linear1 = torch.nn.Linear(num_feat, num_feat)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_feat, n_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
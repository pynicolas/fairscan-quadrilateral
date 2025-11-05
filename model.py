import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Simple Conv + BatchNorm + ReLU block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class QuadRegressorLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(1, 16)
        self.block2 = ConvBlock(16, 32)
        self.block3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2, 2)

        # petit "encoder" global
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.2)

        # régression finale
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # 8 outputs between 0 and 1
        return x

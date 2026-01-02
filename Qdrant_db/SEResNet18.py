
from PyQt6.QtWidgets import QApplication
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision import transforms
from pycocotools.coco import COCO
import numpy as np
from collections import Counter
from torchvision.models.resnet import BasicBlock
import pandas as pd
from CustomDataset import Cocodataset, SampledDataset, SampledDataset_Origin

class SEBlock(nn.Module):
    def __init__(self, channel, reduction="auto"):
        super().__init__()
        if reduction == "auto":
            if channel <= 8:
                r = 1
            elif channel <= 32:
                r = 4
            elif channel <= 128:
                r = 8
            else:
                r = 16
        else:
            r = reduction
        mid = max(1, channel // r)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction="auto"):
        super().__init__(inplanes, planes, stride, downsample)
        self.se = SEBlock(planes, reduction)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)   # 👈 SE
        if self.downsample is not None:
            identity = self.downsample(x) 
        out += identity
        out = self.relu(out)
        return out

class ThinSEResNet18(nn.Module):
    def __init__(self, num_classes, channels=[16, 32, 64, 128], se_reduction="auto"):
        super().__init__()
        assert len(channels) == 4, "channels 必須為 4 個元素"
        self.inplanes = channels[0]
        # Stem
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Stages (2,2,2,2)
        self.layer1 = self._make_layer(SEBasicBlock, channels[0], 2, reduction=se_reduction)
        self.layer2 = self._make_layer(SEBasicBlock, channels[1], 2, stride=2, reduction=se_reduction)
        self.layer3 = self._make_layer(SEBasicBlock, channels[2], 2, stride=2, reduction=se_reduction)
        self.layer4 = self._make_layer(SEBasicBlock, channels[3], 2, stride=2, reduction=se_reduction)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)
    def _make_layer(self, block, planes, blocks, stride=1, reduction="auto"):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, reduction=reduction)
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, reduction=reduction)
            )
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

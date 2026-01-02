import os
import platform
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
import os
from torchvision.models.resnet import BasicBlock
import pandas as pd
num_classes = 11
class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction="auto", spatial_kernel=7):
        super().__init__()

        # ---------- auto reduction（完全沿用你的邏輯） ----------
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

        # ---------- Channel Attention ----------
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channel, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False)
        )

        self.ca_sigmoid = nn.Sigmoid()

        # ---------- Spatial Attention ----------
        self.sa_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False
        )
        self.sa_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # ===== Channel Attention =====
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        ca = self.ca_sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * ca

        # ===== Spatial Attention =====
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.sa_sigmoid(self.sa_conv(sa))
        x = x * sa

        return x
    
class CBAMBasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction="auto", spatial_kernel=7):
        super().__init__(inplanes, planes, stride, downsample)
        self.cbam = CBAMBlock(planes, reduction, spatial_kernel)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)   
        if self.downsample is not None:
            identity = self.downsample(x) 
        out += identity
        out = self.relu(out)
        return out
    
class ThinCBAMResNet18(nn.Module):
    def __init__(self, num_classes, channels=[16, 32, 64, 128], cbam_reduction="auto", spatial_kernel=7):
        super().__init__()
        assert len(channels) == 4, "channels 必須為 4 個元素"
        self.inplanes = channels[0]
        # Stem
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Stages (2,2,2,2)
        self.layer1 = self._make_layer(CBAMBasicBlock, channels[0], 2, reduction=cbam_reduction, spatial_kernel=spatial_kernel)
        self.layer2 = self._make_layer(CBAMBasicBlock, channels[1], 2, stride=2, reduction=cbam_reduction, spatial_kernel=spatial_kernel)
        self.layer3 = self._make_layer(CBAMBasicBlock, channels[2], 2, stride=2, reduction=cbam_reduction, spatial_kernel=spatial_kernel)
        self.layer4 = self._make_layer(CBAMBasicBlock, channels[3], 2, stride=2, reduction=cbam_reduction, spatial_kernel=spatial_kernel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)
    def _make_layer(self, block, planes, blocks, stride=1, reduction="auto", spatial_kernel=7):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, reduction=reduction, spatial_kernel=spatial_kernel)
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, reduction=reduction, spatial_kernel=spatial_kernel)
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

 
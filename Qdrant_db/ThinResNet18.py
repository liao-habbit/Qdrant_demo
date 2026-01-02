import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import BasicBlock

class ThinResNet18(nn.Module):
    def __init__(self, num_classes=1000, channels=None):
        super().__init__()
        # 如果沒給 channels，使用預設 [16,32,64,128]
        if channels is None:
            channels = [16, 32, 64, 128]
        assert len(channels) == 4, "channels 必須有 4 個元素"
        self.inplanes = channels[0]
        # Stem
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNet18 layers (2,2,2,2)
        self.layer1 = self._make_layer(BasicBlock, channels[0], 2)
        self.layer2 = self._make_layer(BasicBlock, channels[1], 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, channels[2], 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, channels[3], 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
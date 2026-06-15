"""
ResNet 模型定义 —— 针对 CIFAR-100 小分辨率优化
改动：
  - 去掉 7x7 conv + MaxPool，改用 3x3 conv（适配 32x32 输入）
  - 末尾 avg_pool2d(4) 对应 4x4 特征图
  - 内置 Kaiming 初始化
"""
import torch.nn as nn
import torch.nn.functional as F


# -------------------- 初始化 --------------------

def _init_weights(m):
    """Kaiming Normal 初始化，适用于 Conv + ReLU 网络"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


# -------------------- 基础模块 --------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel: int, outchannel: int, stride: int = 1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel * self.expansion),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inchannel: int, outchannel: int, stride: int = 1):
        super().__init__()
        internal = outchannel // self.expansion
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, internal, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(internal),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal, internal, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(internal),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal, outchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# -------------------- 网络主体 --------------------

class ResNet_18(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

        # 初始化所有参数
        self.apply(_init_weights)

    def _make_layer(self, block, channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.inchannel, channels, stride=s))
            self.inchannel = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet_50(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(Bottleneck, 256,  3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 512,  4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 1024, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 2048, 3, stride=2)
        self.fc = nn.Linear(2048, num_classes)

        # 初始化所有参数
        self.apply(_init_weights)

    def _make_layer(self, block, channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.inchannel, channels, stride=s))
            self.inchannel = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# -------------------- 工厂函数 --------------------

def get_res18(num_classes: int = 100):
    """创建面向 CIFAR-100 的 ResNet18（从零训练，无预训练权重）"""
    return ResNet_18(num_classes=num_classes)


def get_res50(num_classes: int = 100):
    """创建面向 CIFAR-100 的 ResNet50（从零训练，无预训练权重）"""
    return ResNet_50(num_classes=num_classes)
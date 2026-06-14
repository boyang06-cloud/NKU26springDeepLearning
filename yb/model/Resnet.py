"""
ResNet 模型定义 —— 使用 torchvision 预训练基座，适配 CIFAR-100（100 类）
"""
from torchvision import models
import torch.nn as nn


def get_res18(pretrained=True, num_classes=100):
    """返回适配 CIFAR-100 的 ResNet18"""
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_res50(pretrained=True, num_classes=100):
    """返回适配 CIFAR-100 的 ResNet50"""
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

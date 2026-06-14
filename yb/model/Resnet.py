# 此文件实现res18和res50的模型结构
from torchvision import models

#这里直接用torch官方在imagenet上预训练的模型了，后续在cifar100进行微调
def get_res18():
    return models.resnet18(pretrained=True)


def get_res50():
    return models.resnet50(pretrained=True,weights='ResNet50_Weights.DEFAULT')

print(get_res18())

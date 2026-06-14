"""
工具函数：数据集加载、模型保存/加载、评估指标
"""
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# -------------------- CIFAR-100 数据加载 --------------------

def get_cifar100_dataloaders(batch_size=128, num_workers=2, data_dir='./data'):
    """
    返回 CIFAR-100 的 train / val DataLoader。
    将 32x32 上采样到 224x224 以适配 ImageNet 预训练模型。
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# -------------------- 模型工具 --------------------

def adapt_model_for_cifar100(model, num_classes=100):
    """
    替换模型的最后一层全连接，适配 CIFAR-100 分类数。
    支持 resnet18 / resnet50 等拥有 model.fc 属性的模型。
    """
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(model, optimizer, epoch, best_acc, save_path):
    """保存 checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, save_path)
    print(f'[Save] Checkpoint saved → {save_path}')


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """加载 checkpoint 并返回 epoch 和 best_acc"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'[Load] Checkpoint loaded from → {checkpoint_path}')
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.0)


# -------------------- 评估指标 --------------------

@torch.no_grad()
def evaluate(model, dataloader, device='cuda', num_classes=100):
    """
    在 dataloader 上评估模型，返回：
        top1_acc, top5_acc, precision, recall, f1
    其中 precision / recall / f1 为 macro average。
    """
    model.eval()
    all_preds = []
    all_labels = []
    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # [N, num_classes]

        # Top-1 & Top-5
        _, pred_top5 = outputs.topk(5, dim=1)  # [N, 5]
        top1_correct += (pred_top5[:, 0] == labels).sum().item()
        top5_correct += (pred_top5 == labels.view(-1, 1)).sum().item()
        total += labels.size(0)

        # 收集全部预测（用于 PRF）
        all_preds.extend(pred_top5[:, 0].cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    precision = precision_score(all_labels, all_preds, average='macro',
                                zero_division=0, labels=list(range(num_classes)))
    recall = recall_score(all_labels, all_preds, average='macro',
                          zero_division=0, labels=list(range(num_classes)))
    f1 = f1_score(all_labels, all_preds, average='macro',
                  zero_division=0, labels=list(range(num_classes)))

    return top1_acc, top5_acc, precision, recall, f1


def print_metrics(phase, top1, top5, precision, recall, f1):
    """格式化打印各项指标"""
    print(f'[{phase}] Top-1 Acc:  {top1:.4f}  |  Top-5 Acc:  {top5:.4f}')
    print(f'[{phase}] Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}')

"""
训练 ResNet50 on CIFAR-100
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Resnet import get_res50
from utils import get_cifar100_dataloaders, save_checkpoint, evaluate, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet50 on CIFAR-100')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--data_dir', type=str, default='./data', help='CIFAR-100 data directory')
    parser.add_argument('--log_dir', type=str, default='logs/res50', help='TensorBoard log directory')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/res50', help='checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--num_workers', type=int, default=2, help='dataloader num_workers')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')

    # 数据
    train_loader, val_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )
    print(f'[Data] Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}')

    # 模型
    model = get_res50(pretrained=True, num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(args.log_dir)
    best_acc = 0.0
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        # ---- 训练 ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step()

        # ---- 验证 ----
        top1, top5, precision, recall, f1 = evaluate(model, val_loader, device)

        # ---- TensorBoard ----
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val_top1', top1, epoch)
        writer.add_scalar('Acc/val_top5', top5, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('F1/val', f1, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # ---- 打印 ----
        print(f'\nEpoch [{epoch+1:03d}/{args.epochs:03d}]  '
              f'Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}')
        print_metrics('Val', top1, top5, precision, recall, f1)

        # ---- 保存最佳模型 ----
        if top1 > best_acc:
            best_acc = top1
            save_checkpoint(model, optimizer, epoch, best_acc,
                            os.path.join(args.ckpt_dir, 'best_model.pth'))
            print(f'  >>> New best model saved (Top-1: {best_acc:.4f})')

    writer.close()
    print(f'\n===== Training Complete =====')
    print(f'Best Top-1 Acc: {best_acc:.4f}')


if __name__ == '__main__':
    main()

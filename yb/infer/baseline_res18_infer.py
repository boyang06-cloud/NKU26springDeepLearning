"""
推理评估 ResNet18 on CIFAR-100
计算指标：Top-1 Acc, Top-5 Acc, Precision, Recall, F1
"""
import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Resnet import get_res18
from utils import get_cifar100_dataloaders, load_checkpoint, evaluate, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Inference ResNet18 on CIFAR-100')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to model checkpoint (.pth)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')
    print(f'[Checkpoint] {args.checkpoint}')

    # 数据
    _, val_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )
    print(f'[Data] Val samples: {len(val_loader.dataset)}')

    # 模型
    model = get_res18(pretrained=False, num_classes=100).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # 评估
    top1, top5, precision, recall, f1 = evaluate(model, val_loader, device)

    print('\n' + '=' * 50)
    print('  ResNet18 — CIFAR-100 推理结果')
    print('=' * 50)
    print_metrics('Test', top1, top5, precision, recall, f1)
    print('=' * 50)


if __name__ == '__main__':
    main()

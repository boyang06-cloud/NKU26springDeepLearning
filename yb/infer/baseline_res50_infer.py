"""
推理评估 ResNet50 on CIFAR-100 —— 支持多 GPU
计算指标：Top-1 Acc, Top-5 Acc, Precision, Recall, F1
"""
import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Resnet import get_res50
from utils import get_cifar100_dataloaders, load_checkpoint, evaluate, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Inference ResNet50 on CIFAR-100")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to model checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="GPU indices to use, e.g. --gpu_ids 0 1")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Checkpoint] {args.checkpoint}")

    _, val_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )
    print(f"[Data] Val samples: {len(val_loader.dataset)}")

    model = get_res50(pretrained=False, num_classes=100)
    load_checkpoint(model, args.checkpoint, device)

    if torch.cuda.device_count() > 1:
        print(f"[MultiGPU] Wrap model with DataParallel ({torch.cuda.device_count()} GPUs)")
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    top1, top5, precision, recall, f1 = evaluate(model, val_loader, device)

    print("\n" + "=" * 50)
    print("  ResNet50 — CIFAR-100 推理结果")
    print("=" * 50)
    print_metrics("Test", top1, top5, precision, recall, f1)
    print("=" * 50)


if __name__ == "__main__":
    main()
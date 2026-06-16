import os, sys, argparse, torch, torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.conv2former_raw import conv2former_raw_s
from utils import get_cifar100_dataloaders, load_checkpoint, evaluate, print_metrics

def parse_args():
    p = argparse.ArgumentParser(description="Inference raw Conv2Former-s on CIFAR-100")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gpu_ids", type=int, nargs="+", default=None)
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}  |  Checkpoint: {args.checkpoint}")

    _, val_loader = get_cifar100_dataloaders(args.batch_size, args.num_workers, args.data_dir)
    print(f"[Data] Val samples: {len(val_loader.dataset)}")

    model = conv2former_raw_s(num_classes=100)
    load_checkpoint(model, args.checkpoint, device)
    gpu_count = max(1, torch.cuda.device_count())
    if gpu_count > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    top1, top5, precision, recall, f1 = evaluate(model, val_loader, device)
    print("\n" + "=" * 50)
    print("  Raw Conv2Former-s  --  CIFAR-100 Baseline Inference")
    print("=" * 50)
    print_metrics("Test", top1, top5, precision, recall, f1)
    print("=" * 50)

if __name__ == "__main__":
    main()

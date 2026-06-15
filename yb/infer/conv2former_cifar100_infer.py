import os, sys, argparse, torch, torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.conv2former_small import conv2former_cifar100_t, conv2former_cifar100_s, conv2former_cifar100_b
import torchvision
from utils import get_cifar100_dataloaders, load_checkpoint, evaluate, print_metrics

MODEL_ZOO = {"tiny": conv2former_cifar100_t, "small": conv2former_cifar100_s, "big": conv2former_cifar100_b}

def parse_args():
    p = argparse.ArgumentParser(description="Inference Conv2Former on CIFAR-100")
    p.add_argument("--model", default="small", choices=list(MODEL_ZOO.keys()))
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

    # Use native transforms (not 224x224 upsampling)
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    val_tfm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)])
    val_dataset = torchvision.datasets.CIFAR100(args.data_dir, False, True, val_tfm)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, args.batch_size, False, num_workers=args.num_workers, pin_memory=True)
    print(f"[Data] Val samples: {len(val_dataset)}")

    model = MODEL_ZOO[args.model]()
    load_checkpoint(model, args.checkpoint, device)

    gpu_count = max(1, torch.cuda.device_count())
    if gpu_count > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    top1, top5, precision, recall, f1 = evaluate(model, val_loader, device)
    print("\n" + "=" * 50)
    print(f"  Conv2Former CIFAR-100 ({args.model})  --  Inference Results")
    print("=" * 50)
    print_metrics("Test", top1, top5, precision, recall, f1)
    print("=" * 50)

if __name__ == "__main__":
    main()

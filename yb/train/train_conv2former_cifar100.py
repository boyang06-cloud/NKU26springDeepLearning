import os, sys, argparse, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.conv2former_small import conv2former_cifar100_t, conv2former_cifar100_s, conv2former_cifar100_b
from utils import save_checkpoint, evaluate, print_metrics

MODEL_ZOO = {"tiny": conv2former_cifar100_t, "small": conv2former_cifar100_s, "big": conv2former_cifar100_b}

def get_cifar100_dataloaders(batch_size=128, num_workers=2, data_dir="./data"):
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    val_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR100(data_dir, True, True, train_tfm)
    val_ds   = torchvision.datasets.CIFAR100(data_dir, False, True, val_tfm)
    train_loader = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size, False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def parse_args():
    p = argparse.ArgumentParser(description="Train Conv2Former on CIFAR-100")
    p.add_argument("--model", default="small", choices=["tiny","small","big"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--log_dir", default=None)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gpu_ids", type=int, nargs="+", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gpu_count = max(1, torch.cuda.device_count())
    print(f"[Device] {device}  GPU count: {gpu_count}")

    effective_batch = args.batch_size * gpu_count
    train_loader, val_loader = get_cifar100_dataloaders(effective_batch, args.num_workers, args.data_dir)
    print(f"[Data] Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Batch: {effective_batch}")

    model = MODEL_ZOO[args.model](drop_path_rate=args.drop_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] conv2former_cifar100_{args.model}  Params: {n_params/1e6:.2f}M")
    if gpu_count > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    tag = f"conv2former_{args.model}"
    log_dir = args.log_dir or os.path.join("logs", tag)
    ckpt_dir = args.ckpt_dir or os.path.join("checkpoints", tag)
    os.makedirs(log_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    best_acc = 0.0

    print(f"\n{'='*55}\n  Starting {tag}  {args.epochs} epochs\n  Logs: {log_dir}  Checkpoints: {ckpt_dir}\n{'='*55}\n")

    for epoch in range(args.epochs):
        model.train()
        running_loss = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step()

        top1, top5, precision, recall, f1 = evaluate(model, val_loader, device)

        for k, v in [("Loss/train", train_loss), ("Acc/train", train_acc),
                     ("Acc/val_top1", top1), ("Acc/val_top5", top5),
                     ("Precision/val", precision), ("Recall/val", recall),
                     ("F1/val", f1), ("LR", optimizer.param_groups[0]["lr"])]:
            writer.add_scalar(k, v, epoch)

        print(f"Epoch [{epoch+1:03d}/{args.epochs:03d}]  Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print_metrics("Val", top1, top5, precision, recall, f1)

        if top1 > best_acc:
            best_acc = top1
            net = model.module if hasattr(model, "module") else model
            save_checkpoint(net, optimizer, epoch, best_acc, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"  >>> New best model saved (Top-1: {best_acc:.4f})")

    writer.close()
    print(f"\n{'='*55}\n  Training Complete  |  Best Top-1 Acc: {best_acc:.4f}\n{'='*55}")

if __name__ == "__main__":
    main()

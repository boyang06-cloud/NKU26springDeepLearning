import os, sys, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.conv2former_raw import conv2former_raw_s
from utils import get_cifar100_dataloaders, save_checkpoint, evaluate, print_metrics

def parse_args():
    p = argparse.ArgumentParser(description="Train raw Conv2Former-s baseline on CIFAR-100")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--log_dir", default="logs/conv2former_raw_s")
    p.add_argument("--ckpt_dir", default="checkpoints/conv2former_raw_s")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gpu_ids", type=int, nargs="+", default=None)
    return p.parse_args()

def save_full_checkpoint(model, optimizer, scheduler, epoch, best_acc, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), "best_acc": best_acc,
    }, save_path)
    print(f"  [Save] Checkpoint -> {save_path}")

def main():
    args = parse_args()
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gpu_count = max(1, torch.cuda.device_count())
    print(f"[Device] {device}  GPU count: {gpu_count}")

    effective_batch = args.batch_size * gpu_count
    # Uses 224x224 upsampled dataloader (raw model expects 224x224 input)
    train_loader, val_loader = get_cifar100_dataloaders(effective_batch, args.num_workers, args.data_dir)
    print(f"[Data] Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Batch: {effective_batch}")

    model = conv2former_raw_s(num_classes=100, drop_path_rate=args.drop_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] conv2former_raw_s  Params: {n_params/1e6:.2f}M")
    if gpu_count > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest_ckpt = os.path.join(args.ckpt_dir, "latest_checkpoint.pth")
    best_ckpt = os.path.join(args.ckpt_dir, "best_model.pth")

    writer = SummaryWriter(args.log_dir)
    best_acc = 0.0
    start_epoch = 0
    print(f"\n{'='*55}\n  Starting raw Conv2Former-s baseline  {args.epochs} epochs\n{'='*55}\n")

    try:
        for epoch in range(start_epoch, args.epochs):
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

            print(f"Epoch [{epoch+1:03d}/{args.epochs:03d}]  Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
            print_metrics("Val", top1, top5, precision, recall, f1)

            net = model.module if hasattr(model, "module") else model
            save_full_checkpoint(net, optimizer, scheduler, epoch, best_acc, latest_ckpt)

            if top1 > best_acc:
                best_acc = top1
                save_full_checkpoint(net, optimizer, scheduler, epoch, best_acc, best_ckpt)
                print(f"  >>> New best model saved (Top-1: {best_acc:.4f})")

    except KeyboardInterrupt:
        net = model.module if hasattr(model, "module") else model
        save_full_checkpoint(net, optimizer, scheduler, epoch - 1, best_acc, latest_ckpt)
        print(f"\n[Interrupt] Checkpoint saved. Resume by running again.")
        sys.exit(130)
    except Exception as e:
        net = model.module if hasattr(model, "module") else model
        save_full_checkpoint(net, optimizer, scheduler, epoch - 1, best_acc, latest_ckpt)
        print(f"\n[Error] {e}"); import traceback; traceback.print_exc()
        sys.exit(1)

    writer.close()
    print(f"\n{'='*55}\n  Training Complete  |  Best Top-1 Acc: {best_acc:.4f}\n{'='*55}")

if __name__ == "__main__":
    main()

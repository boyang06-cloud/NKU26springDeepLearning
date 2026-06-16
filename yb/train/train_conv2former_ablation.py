import os, sys, argparse, time, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import evaluate, print_metrics
from model.conv2former_small import LayerNorm, MLP, Block


# ═══════════════════════════════════════════════════════════
#  Ablation: replace Hadamard product with element-wise add
# ═══════════════════════════════════════════════════════════

class SpatialAttentionAdd(nn.Module):
    """Ablation variant: att(x) + v(x) instead of att(x) * v(x)."""
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.att = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.att(x) + self.v(x)  # ⬅ Ablation: + instead of *
        x = self.proj(x)
        return x


class BlockAdd(nn.Module):
    """Same as Block, but uses SpatialAttentionAdd."""
    def __init__(self, index, dim, kernel_size, num_head, window_size=14, mlp_ratio=4., drop_path=0., use_dw=True):
        super().__init__()
        self.attn = SpatialAttentionAdd(dim, kernel_size)
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(dim, mlp_ratio, use_dw=use_dw)
        self.layer_scale_1 = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class Conv2FormerAdd(nn.Module):
    """Conv2Former with Hadamard → Add ablation. Same as Conv2Former but uses BlockAdd."""
    def __init__(self, kernel_sizes=None, img_size=28, in_chans=3, num_classes=100,
                 depths=None, dims=None, window_sizes=None,
                 mlp_ratios=None, num_heads=None,
                 drop_path_rate=0., head_dim=1280, mlp_use_dw=None):
        super().__init__()
        if depths is None:
            depths = [1, 2, 4, 2]
        if dims is None:
            dims = [40, 80, 160, 320]
        if window_sizes is None:
            window_sizes = [14, 14, 7, 4]
        if mlp_ratios is None:
            mlp_ratios = [2, 2, 3, 3]
        if num_heads is None:
            num_heads = [2, 4, 8, 16]

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(dims)

        if mlp_use_dw is None:
            mlp_use_dw = [False, False, True, True]
        assert len(mlp_use_dw) == self.num_stages

        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3, 3]
        assert len(kernel_sizes) == self.num_stages

        # Stem (stride-1 for small inputs)
        self.downsample_layers = nn.ModuleList()
        if img_size < 56:
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]),
                nn.GELU(),
            )
        else:
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GELU(), nn.BatchNorm2d(dims[0] // 2),
                nn.Conv2d(dims[0] // 2, dims[0] // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GELU(), nn.BatchNorm2d(dims[0] // 2),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=2, stride=2, bias=False),
            )
        self.downsample_layers.append(stem)

        for i in range(self.num_stages - 1):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(*[
                BlockAdd(index=cur + j, dim=dims[i], kernel_size=kernel_sizes[i],
                         drop_path=dp_rates[cur + j], num_head=num_heads[i],
                         window_size=window_sizes[i], mlp_ratio=mlp_ratios[i],
                         use_dw=mlp_use_dw[i])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.head = nn.Sequential(
            nn.Conv2d(dims[-1], head_dim, 1),
            nn.GELU(),
            LayerNorm(head_dim, eps=1e-6, data_format="channels_first")
        )
        self.pred = nn.Linear(head_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.head(x)
        return x.mean([-2, -1])

    def forward(self, x):
        return self.pred(self.forward_features(x))


def conv2former_ablation_t(drop_path_rate=0.05, **kwargs):
    """Tiny ablation: Hadamard product → element-wise addition."""
    model = Conv2FormerAdd(
        kernel_sizes=[5, 5, 3, 3], img_size=28, num_classes=100,
        dims=[40, 80, 160, 320], mlp_ratios=[2, 2, 3, 3],
        depths=[1, 2, 4, 2], num_heads=[2, 4, 8, 16],
        mlp_use_dw=[False, False, True, True],
        window_sizes=[14, 14, 7, 4], drop_path_rate=drop_path_rate, head_dim=256, **kwargs)
    return model


# ═══════════════════════════════════════════════════════════
#  Training (same logic as train_conv2former_cifar100.py)
# ═══════════════════════════════════════════════════════════

def get_cifar100_dataloaders(batch_size=128, num_workers=2, data_dir="../data"):
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
    train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=train_tfm)
    val_ds   = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=val_tfm)
    train_loader = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size, False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def save_full_checkpoint(model, optimizer, scheduler, epoch, best_acc, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
    }, save_path)
    print(f"  [Save] Checkpoint -> {save_path}")


def load_full_checkpoint(model, optimizer, scheduler, load_path, device):
    ckpt = torch.load(load_path, map_location=device, weights_only=False)
    try:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"  [Resume] Architecture changed! ({len(missing)} missing, {len(unexpected)} unexpected)")
            return 0, 0.0
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        return start_epoch, best_acc
    except Exception as e:
        print(f"  [Resume] Checkpoint incompatible: {e}")
        return 0, 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Ablation: Conv2Former Hadamard→Add on CIFAR-100")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--drop_path", type=float, default=0.05)
    p.add_argument("--data_dir", default="../data")
    p.add_argument("--log_dir", default=None)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gpu_ids", type=int, nargs="+", default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    return p.parse_args()


def main():
    args = parse_args()
    tag = "conv2former_ablation_t"

    # GPU
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gpu_count = max(1, torch.cuda.device_count())
    print(f"[Device] {device}  GPU count: {gpu_count}")

    # Data
    effective_batch = args.batch_size * gpu_count
    train_loader, val_loader = get_cifar100_dataloaders(effective_batch, args.num_workers, args.data_dir)
    print(f"[Data] Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Batch: {effective_batch}")

    # Model
    model = conv2former_ablation_t(drop_path_rate=args.drop_path)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Ablation-Tiny  Params: {total_params/1e6:.2f}M")
    if gpu_count > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Paths
    log_dir = args.log_dir or os.path.join("logs", tag)
    ckpt_dir = args.ckpt_dir or os.path.join("checkpoints", tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    best_ckpt = os.path.join(ckpt_dir, "best_model.pth")

    # Resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.exists(latest_ckpt):
        net = model.module if hasattr(model, "module") else model
        start_epoch, best_acc = load_full_checkpoint(net, optimizer, scheduler, latest_ckpt, device)
        print(f"[Resume] Loaded {latest_ckpt}")
        print(f"         Resuming epoch {start_epoch+1}/{args.epochs}  |  Best Acc: {best_acc:.4f}")
    elif args.resume and os.path.exists(best_ckpt):
        net = model.module if hasattr(model, "module") else model
        start_epoch, best_acc = load_full_checkpoint(net, optimizer, scheduler, best_ckpt, device)
        print(f"[Resume] Loaded {best_ckpt} (best model, no latest found)")
    else:
        print("[Start] Fresh training (no checkpoint found)")

    writer = SummaryWriter(log_dir)
    print(f"\n{'='*55}")
    print(f"  {tag}  |  Ablation: Hadamard product → Element-wise Addition")
    print(f"  Epochs [{start_epoch+1} -> {args.epochs}]")
    print(f"  Logs: {log_dir}  |  Checkpoints: {ckpt_dir}")
    print(f"{'='*55}\n")

    # Training Loop
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            model.train()
            running_loss = correct = total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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

            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch+1:03d}/{args.epochs:03d}]  "
                  f"Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  "
                  f"Time: {epoch_time:.1f}s")
            print_metrics("Val", top1, top5, precision, recall, f1)

            net = model.module if hasattr(model, "module") else model
            save_full_checkpoint(net, optimizer, scheduler, epoch, best_acc, latest_ckpt)

            if top1 > best_acc:
                best_acc = top1
                save_full_checkpoint(net, optimizer, scheduler, epoch, best_acc, best_ckpt)
                print(f"  >>> New best model saved (Top-1: {best_acc:.4f})")

    except KeyboardInterrupt:
        print(f"\n[Interrupt] Caught Ctrl+C. Saving checkpoint at epoch {epoch+1}...")
        net = model.module if hasattr(model, "module") else model
        save_full_checkpoint(net, optimizer, scheduler, epoch - 1, best_acc, latest_ckpt)
        sys.exit(130)

    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        net = model.module if hasattr(model, "module") else model
        save_full_checkpoint(net, optimizer, scheduler, epoch - 1, best_acc, latest_ckpt)
        sys.exit(1)

    writer.close()
    print(f"\n{'='*55}")
    print(f"  Ablation Training Complete  |  Best Top-1 Acc: {best_acc:.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()

# Conv2Former — CIFAR-100 Adaptation

将 Conv2Former 架构（原论文面向 224×224 ImageNet）适配至 **CIFAR-100（32×32）** 分类任务，包含完整的训练、推理、监控与对比分析工具链。

并且对于改变模型参数，加速训练过程。

## 架构适配要点

| 改动 | 说明 |
|------|------|
| **自适应 Stem** | 输入 < 56px 时使用 stride-1 stem（保留 32×32 分辨率），否则使用原 stride-2 stem |
| **逐 Stage 卷积核** | CIFAR 配置 `[5, 5, 3, 3]`，避免大核在低分辨率下越界 |
| **混合卷积策略** | Stage 0/1 使用密集卷积（高分辨率下 DW conv 效率极低），Stage 2/3 保留深度可分离卷积 |
| **缩减 MLP ratio** | Stage 0/1 的 `mlp_ratio` 从 4 → 2，避免 tall-thin GEMM 瓶颈 |
| **缩减通道/深度** | `dims=[48, 96, 192, 384]`，适配小数据集防止过拟合 |

## 模型变体

| Variant | 参数量 | 配置 |
|---------|--------|------|
| **Tiny** | 3.84M | `dims=[48,96,192,384]`, `depths=[2,2,6,2]`, `head_dim=512` |
| **Small** | 7.44M | `dims=[64,128,256,512]`, `depths=[2,2,6,2]`, `head_dim=512` |
| **Big** | 15.38M | `dims=[64,128,256,512]`, `depths=[2,2,8,2]`, `head_dim=768` |

## 项目结构

```
yb/
├── model/
│   ├── conv2former_small.py       # 适配后的 Conv2Former（推荐使用）
│   ├── conv2former_raw.py         # 原始 Conv2Former 架构（baseline）
│   ├── Resnet.py                  # ResNet-18/50（原始 baseline）
│   └── __init__.py                # 模型导出入口
├── train/
│   ├── train_conv2former_cifar100.py  # 适配模型训练脚本
│   ├── train_conv2former_raw.py       # 原始模型训练脚本
│   ├── train_res18.py                 # ResNet-18 训练
│   ├── train_res50.py                 # ResNet-50 训练
│   ├── *.sh                           # 对应的启动脚本
│   ├── logs/                          # TensorBoard 日志
│   └── checkpoints/                   # 模型权重保存
├── infer/
│   ├── conv2former_cifar100_infer.py  # 适配模型推理评估
│   ├── conv2former_raw_infer.py       # 原始模型推理评估
│   ├── baseline_res18_infer.py        # ResNet-18 推理
│   └── baseline_res50_infer.py        # ResNet-50 推理
├── logs/                          # 其他日志（如 raw baseline）
├── checkpoints/                   # 其他权重
├── data/                          # CIFAR-100 数据集
└── utils.py                       # 工具函数（数据加载、评估指标）
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（推荐）
- 依赖：`timm`, `tensorboard`, `matplotlib`

### 训练（适配模型）

```bash
# 单卡训练 small（推荐）
python -m yb.train.train_conv2former_cifar100 --model small --epochs 100 --batch_size 128

# 指定变体
python -m yb.train.train_conv2former_cifar100 --model tiny --epochs 100
python -m yb.train.train_conv2former_cifar100 --model big   --epochs 100

# 多卡训练
python -m yb.train.train_conv2former_cifar100 --model big --gpu_ids 0 1

# 使用启动脚本
bash yb/train/train_conv2former_cifar100.sh           # 默认 small
bash yb/train/train_conv2former_cifar100.sh tiny       # tiny 变体
bash yb/train/train_conv2former_cifar100.sh big 0 1    # big, 双卡
```

### 训练（原始模型 — baseline）

```bash
python -m yb.train.train_conv2former_raw
```
### 训练消融模型
```
python -m yb.train.train_conv2former_ablation --epochs 100 --batch_size 128 --gpu_ids 0 1 2 3
```
### 推理评估

```bash
# 适配模型
python -m yb.infer.conv2former_cifar100_infer \
    --model small \
    --checkpoint yb/train/checkpoints/conv2former_small/best_model.pth

# 原始 baseline
python -m yb.infer.conv2former_raw_infer \
    --checkpoint yb/checkpoints/conv2former_raw_s/best_model.pth
```

### 训练 ResNet baseline

```bash
cd yb/train
bash train_res18.sh
bash train_res50.sh
```

## 训练监控

```bash
# 实时查看训练指标
tensorboard --logdir yb/train/logs --bind_all

# 支持的监控指标
# - Loss/train       训练损失
# - Acc/train        训练准确率
# - Acc/val_top1     验证集 Top-1 准确率
# - Acc/val_top5     验证集 Top-5 准确率
# - Precision/val    验证集精确率
# - Recall/val       验证集召回率
# - F1/val           验证集 F1 分数
# - LR               学习率
```

## 对比可视化

```bash
# 将三个模型的训练曲线绘制在同一张图上
python plot_comparison.py
```

输出图片保存在项目根目录 `comparison_all.png`，包含 Loss 和 Acc 两个子图。

## 训练机制

- **断点续训**：自动从 `checkpoints/` 目录下的最佳/最新 checkpoint 恢复
- **异常保护**：KeyboardInterrupt（Ctrl+C）时会自动保存当前模型
- **梯度裁剪**：全局梯度范数裁剪到 5.0，防止梯度爆炸
- **Cosine 学习率**：Cosine Annealing 调度，初始 LR=5e-4
- **数据增强**：RandomCrop + RandomHorizontalFlip + Normalize

## 训练速度（RTX 4060 Laptop, batch=128）

| Variant | 1 epoch 耗时 | 100 epochs 预估 |
|---------|-------------|-----------------|
| Tiny    | ~20s        | ~35 min         |
| Small   | ~35s        | ~1 h            |
| Big     | ~90s        | ~2.5 h          |

## 对比结果

| Model | Params | Best Val Acc |
|-------|--------|-------------|
| Conv2Former-Tiny | 3.84M | 33.05% (200 ep) |
| Conv2Former-Small | 7.44M | 31.45% (39 ep) |
| Conv2Former-Big | 15.38M | 36.51% (200 ep) |

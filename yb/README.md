yb/
├── utils.py                          # 工具函数（核心）
├── model/
│   ├── __init__.py                   # 导出 get_res18 / get_res50
│   └── Resnet.py                     # 模型定义（替换 fc 为 100 类）
│   ├── conv2former.py                # Conv2Former 模型定义
├── train/
│   ├── train_res18.py                # ResNet18 训练脚本
│   ├── train_res50.py                # ResNet50 训练脚本
│   ├── train_res18.sh                # 启动脚本
│   └── train_res50.sh                # 启动脚本
│   ├── train_conv2former_cifar100.py # Conv2Former 训练脚本
│   ├── train_conv2former_cifar100.sh # 启动脚本
└── infer/
    ├── baseline_res18_infer.py       # ResNet18 推理评估
    └── baseline_res50_infer.py       # ResNet50 推理评估
    ├── conv2former_cifar100_infer.py # Conv2Former 推理评估

# **quick start**
## 训练resnet:
cd yb/train
bash train_res18.sh

推理评估 :
cd yb
python infer/baseline_res18_infer.py --checkpoint checkpoints/res18/best_model.pth
多gpu推理：
python infer/baseline_res50_infer.py --checkpoint checkpoints/res50/best_model.pth --gpu_ids 0 1

## 训练conv2former:

训练：
```bash
python train_conv2former_cifar100.py --model small --epochs 200 --batch_size 128
python train_conv2former_cifar100.py --model big --gpu_ids 0 1 2 3
```
或者使用脚本训练：
```bash
bash train_conv2former_cifar100.sh           # 默认 small, 单卡
bash train_conv2former_cifar100.sh tiny       # 指定模型为tiny 
bash train_conv2former_cifar100.sh big 0 1    # big, 双卡
```
推理评估：
```bash
python conv2former_cifar100_infer.py --model small --checkpoint checkpoints/conv2former_small/best_model.pth
```

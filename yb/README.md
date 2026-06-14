yb/
├── utils.py                          # 工具函数（核心）
├── model/
│   ├── __init__.py                   # 导出 get_res18 / get_res50
│   └── Resnet.py                     # 模型定义（替换 fc 为 100 类）
├── train/
│   ├── train_res18.py                # ResNet18 训练脚本
│   ├── train_res50.py                # ResNet50 训练脚本
│   ├── train_res18.sh                # 启动脚本
│   └── train_res50.sh                # 启动脚本
└── infer/
    ├── baseline_res18_infer.py       # ResNet18 推理评估
    └── baseline_res50_infer.py       # ResNet50 推理评估

# **quick start**
训练resnet18:
cd yb/train
bash train_res18.sh

推理评估 :
cd yb
python infer/baseline_res18_infer.py --checkpoint checkpoints/res18/best_model.pth
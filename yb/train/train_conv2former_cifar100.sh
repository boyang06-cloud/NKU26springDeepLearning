#!/bin/bash
# Train Conv2Former on CIFAR-100. Supports multi-GPU.
# Usage:
#   bash train_conv2former_cifar100.sh                    # small, single GPU
#   bash train_conv2former_cifar100.sh tiny                # tiny variant
#   bash train_conv2former_cifar100.sh big  0 1 2 3       # big, 4 GPUs

MODEL=${1:-small}
GPUS=${@:2}

CMD="python train_conv2former_cifar100.py --model $MODEL --epochs 200 --batch_size 128"

if [ -n "$GPUS" ]; then
    CMD="$CMD --gpu_ids $GPUS"
    echo "[MultiGPU] GPUs: $GPUS"
fi

echo "[Config] Model: $MODEL"
echo "[Launch] $CMD"
eval $CMD

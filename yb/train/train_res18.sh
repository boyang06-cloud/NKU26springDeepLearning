#!/bin/bash
# ResNet18 训练 —— 可通过 CUDA_VISIBLE_DEVICES 或 --gpu_ids 选择 GPU
# 用法1（默认使用所有可见 GPU）:
#   bash train_res18.sh
# 用法2（指定 GPU）:
#   CUDA_VISIBLE_DEVICES=0,1 bash train_res18.sh
# 用法3（传入 gpu_ids 参数）:
#   bash train_res18.sh --gpu_ids 0 1

# 默认参数
GPU_IDS="--gpu_ids 0 1" #这里 0 1 是默认的 GPU ID，根据实际情况修改
BATCH_SIZE=128
EPOCHS=100
LR=1e-3
WEIGHT_DECAY=5e-4
DATA_DIR="../data"
LOG_DIR="../logs/res18"
CKPT_DIR="../checkpoints/res18"
NUM_WORKERS=2 #根据实际情况修改

# 解析传递给脚本的参数中的 --gpu_ids
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu_ids)
            GPU_IDS="$1"
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                GPU_IDS="$GPU_IDS $1"
                shift
            done
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# 如果设置了 CUDA_VISIBLE_DEVICES，打印出来
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "[Launch] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

CMD="python train_res18.py --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR --weight_decay $WEIGHT_DECAY --data_dir $DATA_DIR --log_dir $LOG_DIR --ckpt_dir $CKPT_DIR --num_workers $NUM_WORKERS"
if [ -n "$GPU_IDS" ]; then
    CMD="$CMD $GPU_IDS"
fi

echo "[Launch] $CMD"
eval $CMD
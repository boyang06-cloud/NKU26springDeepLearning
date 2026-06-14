python train_res50.py \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --data_dir ../data \
    --log_dir ../logs/res50 \
    --ckpt_dir ../checkpoints/res50 \
    --device cuda \
    --num_workers 2

#!/bin/bash

# dataset_dir="/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/IMGUR5K-Handwriting-Dataset"
dataset_dir="/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/tet_gan_20k_psp"
# ckpt_path="/mnt/f06b55a9-977c-474a-bed0-263449158d6a/pixel2style2pixel/pretrained_models/handwriting.pt"
ckpt_path="/mnt/f06b55a9-977c-474a-bed0-263449158d6a/pixel2style2pixel/pretrained_models/042000.pt"
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    --dataset_type "tet_gan" \
    --exp_dir "./experiment" \
    --workers 8 \
    --batch_size 8 \
    --test_batch_size 8 \
    --test_workers 8 \
    --val_interval 2500 \
    --save_interval 5000 \
    --encoder_type GradualStyleEncoder \
    --start_from_latent_avg \
    --lpips_lambda 0.8 \
    --l2_lambda 1 \
    --id_lambda 0 \
    --dataset_dir ${dataset_dir} \
    --user_ckpt_path ${ckpt_path} \
    --use_wandb \
    # --optim_name 'adam' \
    # --learning_rate 0.001\


# python scripts/train.py \
# --dataset_type "hand_writing" \

# --dataset_type=ffhq_encode \
# --exp_dir="./experiment" \
# --workers=8 \
# --batch_size=8 \
# --test_batch_size=8 \
# --test_workers=8 \
# --val_interval=2500 \
# --save_interval=5000 \
# --encoder_type=GradualStyleEncoder \
# --start_from_latent_avg \
# --lpips_lambda=0.8 \
# --l2_lambda=1 \
# --id_lambda=0.1
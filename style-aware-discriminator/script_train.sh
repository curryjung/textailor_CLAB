#!/bin/bash

DEVICE_NUM=0
train_content_dataset_dir="" #train content image path, ex) shadow_datasets/train/mask
train_style_dataset_dir="" #train style image path, ex) shadow_datasets/train/tet_gan
eval_dataset_dir="" #eval dataset path, ex) shadow_datasets/val 
output_dir="" #output path, ex) runs/16pathces_from_2by2_new
exp_discription="some descriptions"

echo "training start"
CUDA_VISIBLE_DEVICES=$DEVICE_NUM python train.py \
    --mod-type adain    \
    --total-nimg 1.6M   \
    --image-size 64     \
    --train_content_dataset $train_content_dataset_dir  \
    --train_style_dataset $train_style_dataset_dir     \
    --eval-dataset $eval_dataset_dir                  \
    --out-dir runs_test/16pathces_from_2by2_new  \
    --extra-desc $exp_discription  \
    --snapshot_freq 200     \
    --eval_freq 200     \
    --save_freq 200     \
    --model_js   \
    --num_patches 64     \
    --intermediate_feat_num 3 \
    --avg_pool \
    --evaluation False     \

echo "done"

#!/bin/bash
d_name_logger="easy_dataset_ocr_modified"
dir_name="easy_dataset_ocr_modified"
# dataset_dir="/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/IMGUR5K-Handwriting-Dataset"
dataset_dir="/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/tet_gan_20k"
ocr_pretrained_model="./OCR/network/TPS-ResNet-BiLSTM-Attn.pth"

CUDA_VISIBLE_DEVICES=0 python train.py --dname_logger ${d_name_logger}\
                        --dataset_dir ${dataset_dir}\
                                --dir_name ${dir_name}\
                                --Transformation TPS \
                                --FeatureExtraction ResNet \
                                --SequenceModeling BiLSTM \
                                --Prediction Attn \
                                --saved_model ./OCR/network/TPS-ResNet-BiLSTM-Attn.pth \
                                --recon_factor 10.0\
                                --wandb\
                                --image_every 100\
                                --iter 50000\
                                --use_ocr_loss_c\
                                --use_ocr_loss_s\
                                --ckpt_every 2000\
                                --dataset 'tet_gen'\
                                --late_ocr_adaptation_n 0\


                                # --get_fixed_gray_text_by_cv2\
                                

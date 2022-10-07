#!/bin/bash
d_name_logger="baseline_with_ocr_modified_ce_from_8000"
dir_name="baseline_with_ocr_modified_ce_from_8000"

python train_renew.py --dname_logger ${d_name_logger}\
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


                                # --get_fixed_gray_text_by_cv2\
                                

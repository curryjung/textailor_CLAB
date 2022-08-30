#!/bin/bash

# Script to run the application

CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python /home/sy/textailor_CLAB/train_encoder_js.py\
        --exp_name "run_js"\
        --exp_disc "overfit_style_encoder_only_batch_8_lr_0.0005"\
        --lr 0.0005\
        --visual_every 3000\
        --save_every 3000\
        --tensorboard


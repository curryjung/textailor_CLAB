#!/bin/bash

# inference results will be save in the same folder as the checkpoint file

DEVICE_NUM=0
checkpoint="" #model checkpoint path
test_content_path="./test/content" #test content image path
test_style_path="./test/style" #test style image path

echo "inference start"

CUDA_VISIBLE_DEVICES=$DEVICE_NUM python -m synthesis swap \
    --checkpoint $checkpoint \
    --folder $test_content_path $test_style_path \

echo "done"
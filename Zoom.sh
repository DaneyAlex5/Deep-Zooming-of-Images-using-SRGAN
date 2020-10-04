#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 Zoom.py \
    --output_dir ./ \
    --summary_dir ./log/ \
    --mode inference \
    --is_training False \
    --task SRGAN \
    --input_dir_LR ./temp/ \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint ./SRGAN_pre-trained/model-200000

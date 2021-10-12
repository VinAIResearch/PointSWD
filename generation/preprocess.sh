#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python generation/preprocess.py \
--config='generation/preprocess_config.json' \
--logdir="logs/"
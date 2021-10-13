#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python classification/preprocess_data.py --config='classification/preprocess_train.json' \
--logdir="logs/"

CUDA_VISIBLE_DEVICES=0 python classification/preprocess_data.py --config='classification/preprocess_test.json' \
--logdir="logs/"
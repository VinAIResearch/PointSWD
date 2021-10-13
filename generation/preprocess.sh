#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python generation/preprocess.py --config='generation/preprocess_train.json' --logdir="logs/"

CUDA_VISIBLE_DEVICES=0 python generation/preprocess.py --config='generation/preprocess_test.json' --logdir="logs/"
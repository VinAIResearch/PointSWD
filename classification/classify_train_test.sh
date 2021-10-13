#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python classification/classification_train.py --config='classification/class_train_config.json' \
--logdir="logs/"

CUDA_VISIBLE_DEVICES=0 python classification/classification_test.py --config='classification/class_test_config.json' \
--logdir="logs/"

#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python generation/preprocess.py --config='generation/preprocess_train.json' --logdir="logs/" --data_path="dataset/shapenet_chair/train.npz"

CUDA_VISIBLE_DEVICES=0 python generation/preprocess.py --config='generation/preprocess_test.json' --logdir="logs/" --data_path="dataset/shapenet_chair/test.npz"

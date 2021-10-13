#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python classification/preprocess_data.py --config='classification/preprocess_train.json' \
--logdir="logs/" \
--data_path="dataset/modelnet40_ply_hdf5_2048/train/"

CUDA_VISIBLE_DEVICES=0 python classification/preprocess_data.py --config='classification/preprocess_test.json' \
--logdir="logs/" \
--data_path="dataset/modelnet40_ply_hdf5_2048/test/"
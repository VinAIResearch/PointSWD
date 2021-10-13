#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python generation/train_latent_generator.py --seed=1 \
--logdir="logs/"

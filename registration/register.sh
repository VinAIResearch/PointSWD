#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python registration/registration_test.py \
--config='registration/registration_config.json' \
--logdir='logs/model/home1/'
CUDA_VISIBLE_DEVICES=0 python registration/registration_test.py \
--config='registration/registration_config.json' \
--logdir='logs/model/home2/'
CUDA_VISIBLE_DEVICES=0 python registration/registration_test.py \
--config='registration/registration_config.json' \
--logdir='logs/model/hotel1/'
CUDA_VISIBLE_DEVICES=0 python registration/registration_test.py \
--config='registration/registration_config.json' \
--logdir='logs/model/hotel2/'
CUDA_VISIBLE_DEVICES=0 python registration/registration_test.py \
--config='registration/registration_config.json' \
--logdir='logs/model/hotel3/'
CUDA_VISIBLE_DEVICES=0 python registration/registration_test.py \
--config='registration/registration_config.json' \
--logdir='logs/model/kitchen/'
CUDA_VISIBLE_DEVICES=0 python registration/registration_test.py \
--config='registration/registration_config.json' \
--logdir='logs/model/lab/'
CUDA_VISIBLE_DEVICES=0 python registration/registration_test.py \
--config='registration/registration_config.json' \
--logdir='logs/model/study/'
echo "DONE register shell"
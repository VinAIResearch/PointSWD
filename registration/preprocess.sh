#!/usr/bin/bash
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--data_path='dataset/home1'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--data_path='dataset/home2'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--data_path='dataset/hotel1'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--data_path='dataset/hotel2'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--data_path='dataset/hotel3'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--data_path='dataset/kitchen'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--data_path='dataset/lab'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--data_path='dataset/study'
echo "DONE preprocess shell"
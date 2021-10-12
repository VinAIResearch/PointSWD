#!/usr/bin/bash
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--dataset='home1'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--dataset='home2'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--dataset='hotel1'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--dataset='hotel2'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--dataset='hotel3'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--dataset='kitchen'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--dataset='lab'
python registration/preprocess_data.py --config='registration/preprocess_config.json' \
--logdir='logs/' \
--dataset='study'
echo "DONE preprocess shell"
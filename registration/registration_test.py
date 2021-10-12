import argparse
import json
import os.path as osp
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from dataset import Fine3dMatchDataset
from matcher import Matcher
from writer import Writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="registration_config.json")
    parser.add_argument("--logdir", type=str, help="folder containing preprocessed data")
    args = parser.parse_args()

    config = args.config
    logdir = args.logdir
    print("Check logdir in 10s: ", logdir)
    time.sleep(10)
    args = json.load(open(config))

    # set seed
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    fname = osp.join(logdir, "config.json")
    with open(fname, "w") as fp:
        json.dump(args, fp, indent=4)

    trans_file = osp.join(logdir, "transformation.log")
    # dataset
    dataset = Fine3dMatchDataset(logdir)
    # matcher
    matcher = Matcher(args["threshold"], args["fitness_threshold"], args["rmse_threshold"])
    # writer
    writer = Writer()
    # process
    num_frag = len(dataset)
    for j in tqdm(range(1, num_frag)):
        for i in tqdm(range(0, j)):
            source_id = int(j)
            target_id = int(i)
            meta_data = "{}  {}  {}\n".format(target_id, source_id, num_frag)

            source_points, source_features = dataset[source_id]
            target_points, target_features = dataset[target_id]
            trans, check = matcher.compute_transformation(
                source_points, target_points, source_features, target_features
            )
            if check:
                writer.write_arr_to_file(trans_file, meta_data, trans)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
from tqdm import tqdm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from dataset.raw3dmatch import ThreeDMatchRawDataset
from models import PointNetAE
from preprocessor import Preprocessor
from utils import load_model_for_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path")
    parser.add_argument("--logdir", type=str, help="log path")
    parser.add_argument("--data_path", type=str, help="home1  home2  hotel1  hotel2  hotel3  kitchen  lab  study")
    args = parser.parse_args()

    config = args.config
    logdir = args.logdir
    print("logdir: ", logdir)
    dset = args.data_path
    args = json.load(open(config))

    # set seed
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    # device
    device = torch.device(args["device"])

    # autoencoder
    if args["autoencoder"] == "pointnet":
        autoencoder = PointNetAE(
            args["embedding_size"],
            args["input_channels"],
            args["output_channels"],
            args["num_points"],
            args["normalize"],
        ).to(device)
    try:
        autoencoder = load_model_for_evaluation(autoencoder, args["model_path"])
    except:
        autoencoder = load_model_for_evaluation(autoencoder, osp.join(logdir, args["model_path"]))

    # dataloader
    # args["root"] = osp.join(args["root"], dset)
    dataset = ThreeDMatchRawDataset(dset)
    # preprocessor
    preprocessor = Preprocessor(autoencoder, device)
    # preprocess
    save_folder = osp.join(logdir, args["save_folder"], args["root"].split("/")[-1])
    if not osp.isdir(save_folder):
        os.makedirs(save_folder)
    for i in tqdm(range(len(dataset))):
        pcd, name = dataset[i]
        points, features = preprocessor.extract_points_and_features(
            pcd, args["voxel_size"], args["radius"], args["num_points"], args["batch_size"], color=args["color"]
        )
        results = {"points": points, "features": features}
        fname = osp.join(save_folder, "{}.npz".format(name))
        np.savez(fname, results)
    print(">save_folder:", save_folder)


if __name__ == "__main__":
    main()

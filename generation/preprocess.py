import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset
from models import PointNetAE
from utils.utils import create_save_folder, initialize_main, load_model_for_evaluation


def main():
    args, logdir = initialize_main()

    # set seed
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    # save_folder
    save_folder = create_save_folder(logdir, args["save_folder"])["save_folder"]

    # device
    device = torch.device(args["device"])

    # network
    if args["architecture"] == "pointnet":
        net = PointNetAE(
            embedding_size=args["embedding_size"],
            input_channels=args["point_channels"],
            output_channels=args["point_channels"],
            num_points=args["num_points"],
            normalize=args["normalize"],
        ).to(device)
    else:
        raise ValueError("Unknown architecture.")

    try:
        net = load_model_for_evaluation(net, args["model_path"])
    except:
        net = load_model_for_evaluation(net, osp.join(logdir, args["model_path"]))

    # dataset
    if args["dataset"] == "shapenetcore55":
        dataset = ShapeNetCore55XyzOnlyDataset(
            args["root"],
            num_points=args["num_points"],
            phase="test",
        )
    else:
        raise ValueError("Unknown dataset type.")

    # dataloader
    dataloader = DataLoader(
        dataset, batch_size=args["batch_size"], shuffle=True, sampler=None, pin_memory=True, worker_init_fn=seed_worker
    )

    # save_path
    save_path = osp.join(save_folder, "latent_codes.npz")

    # main
    latent_vectors_list = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader)):
            pcs = batch.to(device)
            latent_vectors = net.encode(pcs)
            latent_vectors_list.append(latent_vectors.reshape(latent_vectors.shape[0], -1).cpu().numpy())
    latent_vectors_list = np.concatenate(latent_vectors_list, axis=0)
    np.savez(save_path, data=latent_vectors_list)


if __name__ == "__main__":
    main()

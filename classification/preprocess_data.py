import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from add_noise_to_data.random_noise import RandomNoiseAdder
from dataset import ModelNet40
from models import PointCapsNet, PointNetAE
from utils import create_save_folder, initialize_main, load_model_for_evaluation


def main():
    args, logdir = initialize_main()

    # set seed
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    # save_results folder
    save_folder = create_save_folder(logdir, args["save_folder"])["save_folder"]

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

    elif args["autoencoder"] == "pcn":
        autoencoder = PointCapsNet(
            args["prim_caps_size"],
            args["prim_vec_size"],
            args["latent_caps_size"],
            args["latent_vec_size"],
            args["num_points"],
        ).to(device)

    else:
        raise Exception("Unknown autoencoder architecture.")

    try:
        autoencoder = load_model_for_evaluation(autoencoder, args["model_path"])
    except:
        autoencoder = load_model_for_evaluation(autoencoder, osp.join(logdir, args["model_path"]))

    # dataset
    if args["dataset"] == "modelnet40":
        dataset = ModelNet40(args["root"])  # root is a folder containing h5 files
    else:
        raise ValueError("Unknown dataset type.")

    # dataloader
    loader = data.DataLoader(
        dataset,
        batch_size=args["batch_size"],
        pin_memory=args["pin_memory"],
        num_workers=args["num_workers"],
        shuffle=args["shuffle"],
        worker_init_fn=seed_worker,
    )

    # NoiseAdder
    if args["add_noise"]:
        if args["noise_adder"] == "random":
            noise_adder = RandomNoiseAdder(mean=args["mean_noiseadder"], std=args["std_noiseadder"])
        else:
            raise ValueError("Unknown noise_adder type.")

    # save_path
    save_path = osp.join(save_folder, "saved_latent_vectors.npz")

    # main
    latent_vectors_list = []
    labels_list = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(loader)):
            pcs, labels = batch[0].to(device), batch[1].to(device)

            if args["add_noise"]:
                pcs = noise_adder.add_noise(pcs)

            try:
                latent_vectors = autoencoder.encode(pcs)
            except:
                latent_vectors, _ = autoencoder.forward(pcs)
            latent_vectors_list.append(latent_vectors.reshape(latent_vectors.shape[0], -1).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    latent_vectors_list = np.concatenate(latent_vectors_list, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    results = {"latent_vectors": latent_vectors_list, "labels": labels_list}
    np.savez(save_path, **results)


if __name__ == "__main__":
    main()

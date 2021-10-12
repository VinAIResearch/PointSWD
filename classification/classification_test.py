import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from classifier import MLPClassifier
from dataset.modelnet40 import LatentCapsulesModelNet40, LatentVectorsModelNet40
from utils.utils import initialize_main, load_model_for_evaluation


def main():
    args, logdir = initialize_main()

    # set seed
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    # save_results folder
    save_folder = osp.join(logdir, args["save_folder"])
    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    # device
    device = torch.device(args["device"])

    # root
    args["root"] = osp.join(logdir, "latent_codes/model/modelnet40-test/saved_latent_vectors.npz")

    # dataloader
    if args["root"].endswith(".npz"):
        dset = LatentVectorsModelNet40(args["root"])  # root is a npz file
    elif args["root"].endswith(".h5"):
        dset = LatentCapsulesModelNet40(args["root"])
    else:
        raise Exception("Unknown dataset.")
    loader = data.DataLoader(
        dset,
        batch_size=args["batch_size"],
        pin_memory=args["pin_memory"],
        num_workers=args["num_workers"],
        shuffle=args["shuffle"],
        worker_init_fn=seed_worker,
    )

    # classifier
    classifier = MLPClassifier(
        args["input_size"], args["output_size"], args["dropout_p"], [int(i) for i in args["hidden_sizes"].split(",")]
    ).to(device)
    try:
        classifier = load_model_for_evaluation(classifier, args["model_path"])
    except:
        classifier = load_model_for_evaluation(classifier, osp.join(save_folder, args["model_path"]))

    # test main
    num_true = 0
    with torch.no_grad():
        for _, (batch, labels) in tqdm(enumerate(loader)):
            batch = batch.to(device)
            labels = labels.to(device).squeeze().type(torch.long)
            predicted_labels = torch.argmax(F.softmax(classifier(batch), dim=-1), dim=-1)
            num_true += (predicted_labels == labels).sum().item()
    # report
    print("Model: ", save_folder)
    log = "Accuracy: {}\n".format(num_true / len(dset))
    acc = num_true * 1.0 / len(dset)
    print(log)
    with open(os.path.join(save_folder, "accuracy.txt"), "a") as fp:
        fp.write("{} ".format(acc))


if __name__ == "__main__":
    main()

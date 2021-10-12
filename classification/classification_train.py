import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.utils.data as data
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from tqdm import tqdm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from classifier import MLPClassifier
from dataset.modelnet40 import LatentCapsulesModelNet40, LatentVectorsModelNet40
from models.utils import init_weights
from saver import GeneralSaver
from trainer import ClassifierTrainer
from utils.utils import initialize_main


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
    args["root"] = osp.join(logdir, "latent_codes/model/modelnet40-train/saved_latent_vectors.npz")

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

    # optimizer
    if args["optimizer"] == "sgd":
        optimizer = SGD(
            classifier.parameters(),
            lr=args["learning_rate"],
            momentum=args["momentum"],
            weight_decay=args["weight_decay"],
        )
    elif args["optimizer"] == "adam":
        optimizer = Adam(
            classifier.parameters(),
            lr=args["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=args["weight_decay"],
        )
    else:
        raise Exception("Unknown optimizer.")

    # init weights
    if osp.isfile(osp.join(save_folder, args["checkpoint"])):
        print(">Init weights with {}".format(args["checkpoint"]))
        checkpoint = torch.load(osp.join(save_folder, args["checkpoint"]))
        classifier.load_state_dict(checkpoint["classifier"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print(">Init weights with Xavier")
        classifier.apply(init_weights)

    # loss
    loss_func = CrossEntropyLoss()

    # main
    best_loss = args["best_loss"]
    best_epoch = args["best_epoch"]
    start_epoch = int(args["start_epoch"])
    num_epochs = int(args["num_epochs"])
    model_path = osp.join(save_folder, "model.pth")
    train_log = osp.join(save_folder, "train.log")

    classifier.train()
    for epoch in tqdm(range(start_epoch, num_epochs)):
        loss_list = []
        for _, (batch, labels) in tqdm(enumerate(loader)):
            batch = batch.to(device)
            labels = labels.to(device).squeeze().type(torch.long)
            classifier, optimizer, loss = ClassifierTrainer.train(classifier, loss_func, optimizer, batch, labels)
            loss_list.append(loss.item())
        avg_loss = sum(loss_list) / len(loss_list)

        # save checkpoint
        checkpoint_path = osp.join(save_folder, "latest.pth")
        GeneralSaver.save_checkpoint(classifier, optimizer, checkpoint_path, "classifier")

        if epoch % args["epoch_gap_for_save"] == 0:
            checkpoint_path = os.path.join(save_folder, "epoch_" + str(epoch) + ".pth")
            GeneralSaver.save_best_weights(classifier, checkpoint_path)

        # save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            GeneralSaver.save_best_weights(classifier, model_path)

        # report
        log = "Epoch {}| loss: {}\n".format(epoch, avg_loss)
        log_best = "Best epoch {}| best loss: {}".format(best_epoch, best_loss)
        with open(train_log, "a") as fp:
            fp.write(log)
        print(log)
        print(log_best)
        print("---------------------------------------------------------------------------------------")
    # end for
    print("save_folder: ", save_folder)


if __name__ == "__main__":
    main()

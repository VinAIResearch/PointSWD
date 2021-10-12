import os
import os.path as osp
import sys

import numpy as np
from sklearn.svm import LinearSVC
from tqdm import tqdm


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import torch.utils.data as data
from dataset.modelnet40 import LatentCapsulesModelNet40, LatentVectorsModelNet40
from utils.utils import create_save_folder, initialize_main


def main():
    args, logdir = initialize_main()

    # save_results folder
    save_folder = create_save_folder(logdir, args["save_folder"])["save_folder"]

    # datasets
    args["train_root"] = os.path.join(logdir, args["train_root"])
    args["test_root"] = os.path.join(logdir, args["test_root"])

    # train loader
    if args["train_root"].endswith(".npz"):
        train_set = LatentVectorsModelNet40(args["train_root"])  # root is a npz file
    elif args["train_root"].endswith(".h5"):
        train_set = LatentCapsulesModelNet40(args["train_root"])
    else:
        raise Exception("Unknown dataset.")
    train_loader = data.DataLoader(
        train_set,
        batch_size=args["batch_size"],
        pin_memory=args["pin_memory"],
        num_workers=args["num_workers"],
        shuffle=args["shuffle"],
    )

    # test loader
    if args["test_root"].endswith(".npz"):
        test_set = LatentVectorsModelNet40(args["test_root"])  # root is a npz file
    elif args["test_root"].endswith(".h5"):
        test_set = LatentCapsulesModelNet40(args["test_root"])
    else:
        raise Exception("Unknown dataset.")
    test_loader = data.DataLoader(
        test_set,
        batch_size=args["batch_size"],
        pin_memory=args["pin_memory"],
        num_workers=args["num_workers"],
        shuffle=False,
    )

    # classifier
    clf = LinearSVC()

    # main
    train_feature = np.zeros((1, args["input_size"]))
    train_label = np.zeros((1, 1))
    test_feature = np.zeros((1, args["input_size"]))
    test_label = np.zeros((1, 1))
    for batch_id, (latents, labels) in tqdm(enumerate(train_loader)):
        train_label = np.concatenate((train_label, labels.numpy()), axis=None)
        train_label = train_label.astype(int)
        train_feature = np.concatenate((train_feature, latents.numpy()), axis=0)
        if batch_id % 10 == 0:
            print("add train batch: ", batch_id)

    for batch_id, (latents, labels) in tqdm(enumerate(test_loader)):
        test_label = np.concatenate((test_label, labels.numpy()), axis=None)
        test_label = test_label.astype(int)
        test_feature = np.concatenate((test_feature, latents.numpy()), axis=0)
        if batch_id % 10 == 0:
            print("add test batch: ", batch_id)

    train_feature = train_feature[1:, :]
    train_label = train_label[1:]
    test_feature = test_feature[1:, :]
    test_label = test_label[1:]

    print("training the linear SVM.......")
    clf.fit(train_feature, train_label)
    confidence = clf.score(test_feature, test_label)
    print("Accuracy: {} %".format(confidence * 100))

    with open(os.path.join(save_folder, "accuracy.txt"), "a") as fp:
        fp.write(str(confidence) + "\n")


if __name__ == "__main__":
    main()

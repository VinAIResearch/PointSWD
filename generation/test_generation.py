import os.path as osp
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from criteria_comparing_sets_pcs.all_metrics_calculator import AllMetricsCalculator
from criteria_comparing_sets_pcs.jsd_calculator import JsdCalculator
from dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset
from evaluator import SetPcsComparingBasedEvaluator
from generation.train_latent_generator import MLPGenerator
from models import PointNetAE
from utils import create_save_folder, evaluate_on_dataset, initialize_main, load_model_for_evaluation


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

    # test set
    if args["test_set_type"] == "shapenetcore55":
        test_set = ShapeNetCore55XyzOnlyDataset(args["test_root"], args["num_points"], phase="test")

    else:
        raise ValueError("Unknown dataset type.")

    # test loader
    test_loader = DataLoader(
        test_set,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        pin_memory=True,
        shuffle=True,
        worker_init_fn=seed_worker,
    )

    # neural net
    if args["architecture"] == "pointnet":
        model = PointNetAE(
            embedding_size=args["embedding_size"],
            input_channels=args["point_channels"],
            output_channels=args["point_channels"],
            num_points=args["num_points"],
            normalize=args["normalize"],
        ).to(device)
    else:
        raise ValueError("Unknown architecture.")

    try:
        model = load_model_for_evaluation(model, args["model_path"])
    except:
        model = load_model_for_evaluation(model, osp.join(logdir, args["model_path"]))

    # evaluator
    save_file = osp.join(save_folder, "generation_test_results.txt")

    if args["evaluator_type"] == "based_on_comparing_set_pcs":
        evaluator = SetPcsComparingBasedEvaluator()
        # prior_distribution_sampler
        if args["prior_distribution"] == "latent_codes_generator":
            prior_distribution_sampler = MLPGenerator(
                args["latent_dim"], args["n_hidden"], args["hidden_size"], args["embedding_size"]
            ).to(device)

            try:
                prior_distribution_sampler = load_model_for_evaluation(prior_distribution_sampler, args["prior_path"])
            except:
                prior_distribution_sampler = load_model_for_evaluation(
                    prior_distribution_sampler, osp.join(save_folder, args["prior_path"])
                )

        else:
            raise ValueError("Unknown prior distribution.")

        if args["eval_criteria"] == "jsd":
            criteria_calculator = JsdCalculator()
        elif args["eval_criteria"] == "all_metrics":
            criteria_calculator = AllMetricsCalculator()
            args["eval_criteria"] = "jsd"
        else:
            raise ValueError("Unknown eval_criteria")

        eval_dic = {
            "prior_distribution_sampler": prior_distribution_sampler,
            "criteria_calculator": criteria_calculator,
            "batch_size": args["criteria_batch_size"],
            "use_EMD": args["use_EMD"],
            "accelerated_cd": args["accelerated_cd"],
            "save_file": save_file,
        }
    else:
        raise ValueError("Unknown evaluator type.")

    # main
    print("Evaluating...(It might take a while.)")
    avg_eval_value = evaluate_on_dataset(evaluator, model, test_loader, device, **eval_dic)

    # save results
    log = "{}: {}\n".format(args["eval_criteria"], avg_eval_value)
    with open(save_file, "a") as fp:
        fp.write(log)
    print(log)


if __name__ == "__main__":
    main()

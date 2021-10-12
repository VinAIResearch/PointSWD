import os.path as osp
import sys

import torch
import torch.nn as nn


sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from evaluator.evaluator_based_on_comparing_set_pcs.interface import Evaluator


class SetPcsComparingBasedEvaluator(nn.Module):
    def __init__(self):
        super(SetPcsComparingBasedEvaluator, self).__init__()

    @staticmethod
    def evaluate(autoencoder, val_data, prior_distribution_sampler, criteria_calculator, **kwargs):
        autoencoder.eval()
        with torch.no_grad():
            num_samples = val_data.shape[0]
            if kwargs.get("model_type") in ["vae"]:
                _, synthetic_data = autoencoder(val_data)
            else:
                samples = prior_distribution_sampler.sample(num_samples)
                synthetic_data = autoencoder.decode(samples)
            # evaluate
            _evaluation = criteria_calculator.forward(val_data, synthetic_data, **kwargs)
        return {"evaluation": _evaluation}


Evaluator.register(SetPcsComparingBasedEvaluator)
assert issubclass(SetPcsComparingBasedEvaluator, Evaluator)

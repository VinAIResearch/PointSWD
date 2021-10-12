import os.path as osp
import sys

import torch.nn as nn


sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from criteria_comparing_sets_pcs.jsd_calculator import JsdCalculator


class JSDBasedEvaluator(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def evaluate(autoencoder, val_data, prior_distribution_sampler, **kwargs):
        num_samples = val_data.shape[0]
        samples = prior_distribution_sampler.sample(num_samples)
        synthetic_data = autoencoder.decode(samples)
        JSD_evaluation = JsdCalculator.forward(val_data, synthetic_data, **kwargs)
        return {"evaluation": JSD_evaluation}

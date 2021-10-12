import os.path as osp
import sys

import torch
import torch.nn as nn


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from metrics_from_point_flow.evaluation_metrics import jsd_between_point_cloud_sets


class JsdCalculator(nn.Module):
    def __init__(self):
        super(JsdCalculator, self).__init__()

    @staticmethod
    def forward(sample_pcs, ref_pcs, resolution=28, **kwargs):
        sample_pcs = sample_pcs.detach().cpu().numpy()
        ref_pcs = ref_pcs.detach().cpu().numpy()
        return jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution)


if __name__ == "__main__":
    sample_pcs = torch.empty(5, 2048, 3).uniform_(0, 1).numpy()
    ref_pcs = torch.empty(5, 2048, 3).uniform_(0, 1).numpy()
    print(JsdCalculator.forward(sample_pcs, ref_pcs))

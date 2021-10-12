import os.path as osp
import sys

import torch.nn as nn


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from metrics_from_point_flow.evaluation_metrics import emd_approx


class EMD(nn.Module):
    """
    Auction algorithm to estimate EMD was proposed in paper "A distributed algorithm for the assignment problem" -  Dimitri P. Bertsekas, 1979.
    """

    def __init__(self, *args, **kwargs):
        super(EMD, self).__init__()

    def forward(self, x, y, **kwargs):
        """
        x, y: [batch size, num points in point cloud, 3]
        """
        return {"loss": emd_approx(x, y).mean()}

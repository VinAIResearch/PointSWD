import os.path as osp
import sys

import torch
import torch.nn as nn


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from criteria_comparing_sets_pcs.jsd_calculator import JsdCalculator
from metrics_from_point_flow.evaluation_metrics import compute_all_metrics


class AllMetricsCalculator(nn.Module):
    def __init__(self):
        super(AllMetricsCalculator, self).__init__()

    @staticmethod
    def forward(sample_pcs, ref_pcs, batch_size, **kwargs):
        results = {}
        results.update(compute_all_metrics(sample_pcs, ref_pcs, batch_size, **kwargs))
        for key, value in results.items():
            if torch.is_tensor(value):
                results[key] = value.item()
        if "save_file" in kwargs.keys():
            log = "{}: {}\n"
            with open(kwargs["save_file"], "a") as fp:
                for key, value in results.items():
                    fp.write(log.format(key, value))
                # end for
            # end with
        # end if
        print("\n")
        log = "{}: {}\n"
        for key, value in results.items():
            print(log.format(key, value))
        # end for
        jsd = JsdCalculator.forward(sample_pcs, ref_pcs, **kwargs)
        return jsd


if __name__ == "__main__":
    sample_pcs = torch.empty(10, 2048, 3).uniform_(0, 1).cuda()
    ref_pcs = torch.empty(10, 2048, 3).uniform_(0, 1).cuda()
    batch_size = 10
    print(AllMetricsCalculator.forward(sample_pcs, ref_pcs, batch_size))

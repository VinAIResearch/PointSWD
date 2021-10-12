import os.path as osp
import sys
sys.path.append(osp.dirname( osp.dirname( osp.abspath(__file__) ) ))

import torch
import torch.nn as nn

from criteria_comparing_sets_pcs.evaluation_metrics import minimum_mathing_distance

class MmdCalculator(nn.Module):
    def __init__(self):
        super(MmdCalculator, self).__init__()
    
    @staticmethod
    def forward(sample_pcs, ref_pcs, batch_size, use_EMD=True, **kwargs):
        try:
            sample_pcs = sample_pcs.detach().cpu().numpy()
            ref_pcs = ref_pcs.detach().cpu().numpy()
        except:
            sample_pcs = sample_pcs.cpu().numpy()
            ref_pcs = ref_pcs.cpu().numpy()   
        return minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, use_EMD=use_EMD, **kwargs)["mmd"]
if __name__ == "__main__":
    import numpy as np
    import time

    batch_size = 64

    sample_pcs = torch.empty(batch_size, 2048, 3).uniform_(0,1).numpy()
    ref_pcs = torch.empty(batch_size, 2048, 3).uniform_(0,1).numpy()
    
    start_time = time.time()
    MmdCalculator.forward(sample_pcs, ref_pcs, batch_size)
    print("time: ", time.time() - start_time)
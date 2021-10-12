import os.path as osp
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def minibatch_rand_projections(batchsize, dim, num_projections=1000, **kwargs):
    projections = torch.randn((batchsize, num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=2, keepdim=True))
    return projections


def proj_onto_unit_sphere(vectors):
    """
    input: vectors: [batchsize, num_projs, dim]
    """
    return vectors / torch.sqrt(torch.sum(vectors ** 2, dim=2, keepdim=True))


def _sample_minibatch_orthogonal_projections(batch_size, dim, num_projections):
    projections = torch.zeros((batch_size, num_projections, dim))
    projections = torch.stack([torch.nn.init.orthogonal_(projections[i]) for i in range(projections.shape[0])], dim=0)
    return projections


def compute_practical_moments_sw(x, y, num_projections=30, device="cuda", degree=2.0, **kwargs):
    """
    x, y: [batch_size, num_points, dim=3]
    num_projections: integer number
    """
    dim = x.size(2)
    batch_size = x.size(0)
    projections = minibatch_rand_projections(batch_size, dim, num_projections).to(device)
    # projs.shape: [batchsize, num_projs, dim]

    xproj = x.bmm(projections.transpose(1, 2)).to(device)

    yproj = y.bmm(projections.transpose(1, 2)).to(device)

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0]).to(device)

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = _sort_pow_p_get_sum.mean(dim=1)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def compute_practical_moments_sw_with_predefined_projections(x, y, projections, device="cuda", degree=2.0, **kwargs):
    """
    x, y: [batch size, num points, dim]
    projections: [batch size, num projs, dim]
    """
    xproj = x.bmm(projections.transpose(1, 2)).to(device)

    yproj = y.bmm(projections.transpose(1, 2)).to(device)

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0]).to(device)

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = _sort_pow_p_get_sum.mean(dim=1)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def _compute_practical_moments_sw_with_projected_data(xproj, yproj, device="cuda", degree=2.0, **kwargs):
    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0]).to(device)

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = _sort_pow_p_get_sum.mean(dim=1)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def _circular(x, theta):
    """The circular defining function for generalized Radon transform
    Inputs
    X:  [batch size, num_points, d] - d: dim of 1 point
    theta: [batch size, L, d] that parameterizes for L projections
    """
    x_s = torch.stack([x for _ in range(theta.shape[1])], dim=2)
    theta_s = torch.stack([theta for _ in range(x.shape[1])], dim=1)
    z_s = x_s - theta_s
    return torch.sqrt(torch.sum(z_s ** 2, dim=3))


def _linear(x, theta):
    """
    x: [batch size, num_points, d] - d: dim of 1 point
    theta: [batch size, L, d] that parameterizes for L projections
    """
    xproj = x.bmm(theta.transpose(1, 2))
    return xproj


class SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        squared_sw_2, _ = compute_practical_moments_sw(x, y, num_projections=self.num_projs, device=self.device)
        return {"loss": squared_sw_2.mean(dim=0)}


class ASW(nn.Module):
    """
    Adaptive sliced wasserstein algorithm for estimating SWD
    """

    def __init__(
        self,
        init_projs=2,
        step_projs=1,
        k=2.0,
        loop_rate_thresh=0.05,
        projs_history="projs_history.txt",
        max_slices=500,
        **kwargs
    ):
        super().__init__()
        self.init_projs = init_projs
        self.step_projs = step_projs
        self.k = k
        self.loop_rate_thresh = loop_rate_thresh
        self.projs_history = projs_history
        self.max_slices = max_slices
        if "device" in kwargs.keys():
            self.device = kwargs["device"]
        else:
            self.device = "cuda"

    def forward(self, x, y, **kwargs):
        """
        x, y: [batch size, num points in point cloud, 3]
        """
        # allow to adjust epsilon
        if "epsilon" in kwargs.keys():
            epsilon = kwargs["epsilon"]
        else:
            raise ValueError("Epsilon not found.")

        n = self.init_projs
        max_slices = self.max_slices
        step_projs = self.step_projs

        first_moment_sw_p_pow_p, second_moment_sw_p_pow_p = compute_practical_moments_sw(
            x, y, num_projections=n, device=self.device, degree=kwargs["degree"]
        )

        loop_conditions = (self.k ** 2 * (second_moment_sw_p_pow_p - first_moment_sw_p_pow_p ** 2)) > (
            (n - 1) * epsilon ** 2
        )  # check ASW condition
        loop_rate = (
            loop_conditions.sum(dim=0) * 1.0 / loop_conditions.shape[0]
        )  # the ratio of point clouds in the batch satifying the ASW condition.

        while (loop_rate > self.loop_rate_thresh) and ((n + step_projs) <= max_slices):

            first_moment_s_sw, second_moment_s_sw = compute_practical_moments_sw(
                x, y, num_projections=step_projs, device=self.device, degree=kwargs["degree"]
            )  # sample next s projections

            first_moment_sw_p_pow_p = (n * first_moment_sw_p_pow_p + step_projs * first_moment_s_sw) / (
                n + step_projs
            )  # update first and second moments
            second_moment_sw_p_pow_p = (n * second_moment_sw_p_pow_p + step_projs * second_moment_s_sw) / (
                n + step_projs
            )
            n = n + step_projs
            loop_conditions = (self.k ** 2 * (second_moment_sw_p_pow_p - first_moment_sw_p_pow_p ** 2)) > (
                (n - 1) * epsilon ** 2
            )
            loop_rate = loop_conditions.sum(dim=0) * 1.0 / loop_conditions.shape[0]

        with open(self.projs_history, "a") as fp:  # jot down number of sampled projections
            fp.write(str(n) + "\n")
        return {"loss": first_moment_sw_p_pow_p.mean(dim=0), "num_slices": n}


class MaxSW(nn.Module):
    """
    Max-SW distance was proposed in paper "Max-Sliced Wasserstein Distance and its use for GANs" - CVPR'19
    The way to estimate it was proposed in paper "Generalized Sliced Wasserstein Distance" - NeurIPS'19
    """

    def __init__(self, device="cuda", **kwargs):
        super().__init__()
        self.device = device

    def forward(self, x, y, *args, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        dim = x.size(2)
        projections = Variable(
            minibatch_rand_projections(batchsize=x.size(0), dim=dim, num_projections=1).to(self.device),
            requires_grad=True,
        )
        # projs.shape: [batchsize, num_projs, dim]

        num_iter = kwargs.get("max_sw_num_iters") if "max_sw_num_iters" in kwargs.keys() else 50
        lr = kwargs.get("max_sw_lr") if "max_sw_lr" in kwargs else 1e-4
        optimizer = torch.optim.Adam([projections], lr=lr)

        for _ in range(num_iter):
            # compute loss
            xproj = x.bmm(projections.transpose(1, 2)).to(self.device)

            yproj = y.bmm(projections.transpose(1, 2)).to(self.device)

            _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0]).to(self.device)

            _sort_pow_2_get_sum = torch.sum(torch.pow(_sort, 2), dim=2)

            negative_first_moment = -_sort_pow_2_get_sum.mean(dim=1)

            # perform optimization
            optimizer.zero_grad()
            negative_first_moment.mean().backward(retain_graph=True)
            optimizer.step()
            # project onto unit sphere
            projections = proj_onto_unit_sphere(projections)

        projections_no_grad = projections.detach()
        loss, _ = compute_practical_moments_sw_with_predefined_projections(x, y, projections_no_grad, self.device)
        loss = loss.mean(dim=0)

        return {"loss": loss}


class OrtSW(nn.Module):
    """
    Orthogonal estimation for SWD was proposed in paper "Orthogonal estimation of Wasserstein Distance - AISTATS'19"
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        projections = torch.zeros(
            (x.shape[0], self.num_projs, x.shape[2]), dtype=x.dtype, layout=x.layout, device=x.device
        )

        projections = torch.stack(
            [torch.nn.init.orthogonal_(projections[i]) for i in range(projections.shape[0])], dim=0
        )

        loss, _ = compute_practical_moments_sw_with_predefined_projections(x, y, projections, device=self.device)

        return {"loss": loss.mean(dim=0)}


class GenSW(nn.Module):
    """
    Generalized SW distance was proposed in paper "Generalized Sliced Wasserstein Distance" - NeurIPS'19
    """

    def __init__(self, num_projs, g_type="circular", device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device
        self.g_type = g_type

    def forward(self, x, y, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        dim = x.size(2)
        batch_size = x.size(0)
        projections = minibatch_rand_projections(batch_size, dim, self.num_projs).to(self.device)

        if self.g_type == "circular":
            xproj = _circular(x, projections)
            yproj = _circular(y, projections)
        elif self.g_type == "linear":
            xproj = _linear(x, projections)
            yproj = _linear(y, projections)
        else:
            raise NotImplementedError

        loss, _ = _compute_practical_moments_sw_with_projected_data(xproj, yproj, self.device, kwargs["degree"])

        return {"loss": loss.mean(dim=0)}


class PW(nn.Module):
    """
    Projected Wasserstein distance was proposed in paper "Orthogonal estimation of Wasserstein Distance - AISTATS'19"
    """

    def __init__(self, num_projs, device="cuda", orthogonal=False, **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device
        self.orthogonal = orthogonal

    def forward(self, x, y, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """

        dim = x.size(2)
        batch_size = x.size(0)
        if self.orthogonal:
            projections = _sample_minibatch_orthogonal_projections(batch_size, dim, self.num_projs).to(self.device)
        else:
            projections = minibatch_rand_projections(batch_size, dim, self.num_projs).to(self.device)
        # print(projections)
        xproj = _linear(x, projections).transpose(1, 2)  # [bs, num_slices, num_points]
        yproj = _linear(y, projections).transpose(1, 2)  # [bs, num_slices, num_points]

        xproj_argsort = torch.argsort(xproj, dim=2)
        yproj_argsort = torch.argsort(yproj, dim=2)

        _sorted_x = torch.stack([x[i][xproj_argsort[i]] for i in range(x.shape[0])], dim=0)
        _sorted_y = torch.stack([y[i][yproj_argsort[i]] for i in range(y.shape[0])], dim=0)

        loss = torch.mean((_sorted_x - _sorted_y) ** 2)
        return {"loss": loss}

import random

import numpy as np
import torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


import warnings

# deepul borrowed from https://github.com/rll/deepul
import deepul.pytorch_util as ptu


warnings.filterwarnings("ignore")

ptu.set_gpu_mode(True)

import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from scipy.stats import norm
from torch.autograd import Variable
from tqdm import tqdm


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils.utils import create_save_folder


class L2MatrixComputer(nn.Module):
    def __init__(self):
        super(L2MatrixComputer, self).__init__()

    @staticmethod
    def compute_cost_matrix(data, generated_data, **kwargs):
        """
        data, generated_data: shape [batch_size, dim]
        """
        x_col = data.unsqueeze(1)
        y_lin = generated_data.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** 2, 2)
        if ("scale" in kwargs.keys()) and (torch.max(c).item() > 100):
            return kwargs["scale"] * c
        else:
            return c


def sink_stabilized(M, reg, numItermax=1000, tau=1e2, stopThr=1e-9, warmstart=None, print_period=20, cuda=True):

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if cuda:
            alpha, beta = Variable(torch.zeros(na)).cuda(), Variable(torch.zeros(nb)).cuda()
        else:
            alpha, beta = Variable(torch.zeros(na)), Variable(torch.zeros(nb))
    else:
        alpha, beta = warmstart

    if cuda:
        u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
    else:
        u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

    def get_K(alpha, beta):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        return torch.exp(
            -(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg
            + torch.log(u.view((na, 1)))
            + torch.log(v.view((1, nb)))
        )

    # print(np.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        v = torch.div(b, (K.t().matmul(u) + 1e-16))
        u = torch.div(a, (K.matmul(v) + 1e-16))

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)).item() > tau or torch.max(torch.abs(v)).item() > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)

            if cuda:
                u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
            else:
                u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            transp = get_Gamma(alpha, beta, u, v)
            err = (torch.sum(transp) - b).norm(1).pow(2).item()

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print("Warning: numerical errors at iteration", cpt)
            u = uprev
            v = vprev
            break

        cpt += 1

    return torch.sum(get_Gamma(alpha, beta, u, v) * M)


def sinkhorn(gt, pr, epsilon):
    cost_matrix = L2MatrixComputer.compute_cost_matrix(gt, pr)
    return sink_stabilized(cost_matrix, epsilon)


def savefig(fname, show_figure=True):
    if not osp.exists(osp.dirname(fname)):
        os.makedirs(osp.dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def plot_gan_training(losses, title, fname):
    plt.figure()
    n_itr = len(losses)
    xs = np.arange(n_itr)

    plt.plot(xs, losses, label="loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    savefig(fname)


def train(
    generator,
    critic,
    c_loss_fn,
    g_loss_fn,
    train_loader,
    g_optimizer,
    c_optimizer,
    n_critic=1,
    g_scheduler=None,
    c_scheduler=None,
    weight_clipping=None,
):
    """
    generator:
    critic: discriminator in 1ab, general model otherwise
    loss_fn
    train_loader: instance of DataLoader class
    optimizer:
    ncritic: how many critic gradient steps to do for every generator step
    """
    g_losses, c_losses = [], []
    generator.train()
    critic.train()
    for i, x in enumerate(train_loader):
        x = x.to(ptu.device).float()
        c_loss = c_loss_fn(generator, critic, x)
        c_optimizer.zero_grad()
        c_loss.backward()
        c_optimizer.step()
        c_losses.append(c_loss.item())
        if weight_clipping is not None:
            for param in critic.parameters():
                param.data.clamp_(-weight_clipping, weight_clipping)

        if i % n_critic == 0:  # generator step
            g_loss = g_loss_fn(generator, critic, x)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            g_losses.append(g_loss.item())
            if g_scheduler is not None:
                g_scheduler.step()
            if c_scheduler is not None:
                c_scheduler.step()
    return dict(g_losses=g_losses, c_losses=c_losses)


def train_epochs(generator, critic, g_loss_fn, c_loss_fn, train_loader, train_args):
    epochs, lr = train_args["epochs"], train_args["lr"]
    if "optim_cls" in train_args:
        g_optimizer = train_args["optim_cls"](generator.parameters(), lr=lr)
        c_optimizer = train_args["optim_cls"](critic.parameters(), lr=lr)
    else:
        g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0, 0.9))
        c_optimizer = optim.Adam(critic.parameters(), lr=lr, betas=(0, 0.9))

    if train_args.get("lr_schedule", None) is not None:
        g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, train_args["lr_schedule"])
        c_scheduler = optim.lr_scheduler.LambdaLR(c_optimizer, train_args["lr_schedule"])
    else:
        g_scheduler = None
        c_scheduler = None

    train_losses = dict()
    for epoch in tqdm(range(epochs), desc="Epoch", leave=False):
        generator.train()
        critic.train()
        train_loss = train(
            generator,
            critic,
            c_loss_fn,
            g_loss_fn,
            train_loader,
            g_optimizer,
            c_optimizer,
            n_critic=train_args.get("n_critic", 0),
            g_scheduler=g_scheduler,
            c_scheduler=c_scheduler,
            weight_clipping=train_args.get("weight_clipping", None),
        )

        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
            train_losses[k].extend(train_loss[k])

    return {"train_losses": train_losses, "generator": generator}


def get_training_snapshot(generator, critic, n_samples=5000):
    generator.eval()
    critic.eval()
    xs = np.linspace(-1, 1, 1000)
    samples = ptu.get_numpy(generator.sample(n_samples))
    critic_output = ptu.get_numpy(critic(ptu.FloatTensor(xs).unsqueeze(1)))
    return samples, xs, critic_output


class MLP(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU(0.2))
            input_size = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLPGenerator(nn.Module):
    def __init__(self, latent_dim, n_hidden, hidden_size, data_dim):
        super().__init__()
        layers = []
        self.latent_dim = latent_dim
        self.mlp = MLP(latent_dim, n_hidden, hidden_size, data_dim)

    def forward(self, z):
        return torch.tanh(self.mlp(z))

    def sample(self, n):
        # n is the number of samples to return
        z = ptu.normal(ptu.zeros(n, self.latent_dim), ptu.ones(n, self.latent_dim))
        return self.forward(z)


class MLPDiscriminator(nn.Module):
    def __init__(self, latent_dim, n_hidden, hidden_size, data_dim):
        super().__init__()
        self.mlp = MLP(latent_dim, n_hidden, hidden_size, data_dim)

    def forward(self, z):
        return torch.sigmoid(self.mlp(z))


def q1_b(train_data):  # train OT
    """
    train_data: An (5763, 256) numpy array: shapenet chair training data
    """
    # create data loaders
    loader_args = dict(batch_size=64, shuffle=True, worker_init_fn=seed_worker)
    train_loader = data.DataLoader(train_data, **loader_args)

    # model
    g = MLPGenerator(64, 3, 128, 256).to(ptu.device)
    c = MLPDiscriminator(256, 3, 128, 1).to(ptu.device)

    # loss functions
    def g_loss_2(generator, critic, x):  # have no effect on discriminator
        fake_data = generator.sample(x.shape[0])
        loss = sinkhorn(x, fake_data, 0.001)
        print("loss: ", loss.item())
        return loss

    def c_loss_2(generator, critic, x):  # have no effect on generator
        fake_data = torch.empty(x.shape).uniform_(0, 1).cuda()
        return -(1 - critic(fake_data)).log().mean() - critic(x).log().mean()

    # train
    # g_loss_2, c_loss_2: sinkhorn 1e-3
    result_dic = train_epochs(g, c, g_loss_2, c_loss_2, train_loader, dict(epochs=25, lr=1e-4, n_critic=1, q1=True))
    train_losses = result_dic["train_losses"]
    generator = result_dic["generator"]

    return {"g_losses": train_losses["g_losses"], "generator": generator}


def save_results(part, fn, save_folder):
    npz_latent_codes_path = osp.join(save_folder, "latent_codes.npz")  ##path to npz file
    data = np.load(npz_latent_codes_path)["data"]
    result_dict = fn(data)
    losses = result_dict["g_losses"]
    generator = result_dict["generator"]

    generator_save_path = osp.join(save_folder, "generator.pth")
    torch.save(generator.state_dict(), generator_save_path)

    # loss plot
    plot_gan_training(losses, "Q1{} Losses".format(part), osp.join(save_folder, "q1{}_losses.png".format(part)))
    print(">Plot at:", osp.join(save_folder, "q1{}_losses.png".format(part)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    logdir = args.logdir

    save_folder = "shapenet_chair/train/"
    save_folder = create_save_folder(logdir, save_folder)["save_folder"]

    save_results("b", q1_b, save_folder)

import argparse
import datetime
import math
import os
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class LinearLayer(torch.nn.Module):
    def __init__(self, m_in, m_out, m, q, a, b):
        super().__init__()
        self.phi = lambda s: a * s + b * torch.abs(s)
        self.d_phi = lambda s: a + b * torch.sign(s)
        self.theta = torch.nn.Parameter(
            torch.normal(
                mean=0.0,
                std=(a ** 2 + b ** 2) ** (- 1 / 2) * m ** (-q / 2),
                size=(m_out, m_in)
            )
        )
        self.theta_multiplier = m ** (q / 2)
        self.premultiplier = m_out ** (1 / 2)
        self.postmultiplier = m_out ** (-1 / 2)

    def fwd(self, theta, f):
        return self.postmultiplier * self.phi(
            self.premultiplier * self.postmultiplier * self.theta_multiplier * F.linear(f, theta)
        )

    def forward(self, f):
        self.f_prev = f
        self.theta_f_prev = F.linear(self.f_prev, self.theta)
        self.ff = self.postmultiplier * self.theta_multiplier * self.theta_f_prev
        return self.postmultiplier * self.phi(self.premultiplier * self.ff)

    def backward(self, theta_t_b_next):
        self.bb = self.postmultiplier * self.theta_multiplier * theta_t_b_next
        self.b = self.d_phi(self.premultiplier * self.ff) * self.bb
        n = self.f_prev.size(0)
        self.theta.grad = (self.b.t() * n ** (-1 / 2)) @ (self.f_prev * n ** (-1 / 2))
        self.theta_t_b = F.linear(self.b, self.theta.t())
        return self.theta_t_b

    def post_backward(self, p):
        n = self.f_prev.size(0)
        self.F_prev = self.f_prev @ self.f_prev.t() / n
        self.B = self.b @ self.b.t() / n
        self.F_prev_fr = torch.linalg.matrix_norm(self.F_prev, ord='fro')
        self.F_prev_tr = torch.trace(self.F_prev)
        self.B_fr = torch.linalg.matrix_norm(self.B, ord='fro')
        self.B_tr = torch.trace(self.B)
        self.grad_scale = self.F_prev_tr ** (-p / 2) * self.B_tr ** (-p / 2)

    def scale_grad(self):
        self.theta.grad *= self.grad_scale


class ReadoutLayer(torch.nn.Module):
    def __init__(self, m_in, m_out, m, q, a, b):
        super().__init__()
        self.theta = torch.nn.Parameter(
            torch.normal(
                mean=0.0,
                std=(a ** 2 + b ** 2) ** (- 1 / 2) * m ** (-q / 2),
                size=(m_out, m_in)
            )
        )

    def fwd(self, theta, f):
        return F.linear(f, theta)

    def forward(self, f):
        self.f_prev = f
        self.theta_f_prev = F.linear(self.f_prev, self.theta)
        return self.theta_f_prev

    def backward(self, b):
        self.b = b
        n = self.f_prev.size(0)
        self.theta.grad = (self.b.t() * n ** (-1 / 2)) @ (self.f_prev * n ** (-1 / 2))
        self.theta_t_b = F.linear(self.b, self.theta.t())
        return self.theta_t_b

    def post_backward(self, p):
        self.F_prev = self.f_prev @ self.f_prev.t() / n
        self.B = self.b @ self.b.t() / n
        self.F_prev_fr = torch.linalg.matrix_norm(self.F_prev, ord='fro')
        self.F_prev_tr = torch.trace(self.F_prev)
        self.B_fr = torch.linalg.matrix_norm(self.B, ord='fro')
        self.B_tr = torch.trace(self.B)
        self.grad_scale = self.F_prev_tr ** (-p / 2) * self.B_tr ** (-p / 2)

    def scale_grad(self):
        self.theta.grad *= self.grad_scale


class NUMLP(nn.Module):
    def __init__(self, m_0, m_l, m, l, a, b, q, r):
        super(NUMLP, self).__init__()
        ms = [m_0] + [(l - k) ** r * m for k in range(l)] + [m_l]
        self.layers = nn.ModuleList(
            [LinearLayer(ms[k], ms[k + 1], m, q, a, b) for k in range(l)] +
            [ReadoutLayer(ms[l], ms[l + 1], m, q, a, b)]
        )

    def theta(self):
        return tuple(layer.theta.detach() for layer in self.layers)

    def nabla(self):
        return tuple(layer.theta.grad.detach() for layer in self.layers)

    def scaled_nabla(self):
        return tuple(layer.grad_scale * layer.theta.grad.detach() for layer in self.layers)

    def fwd(self, theta, f):
        for layer_k, theta_k in zip(self.layers, theta):
            f = layer_k.fwd(theta_k, f)
        return f

    def forward(self, f):
        for layer in self.layers:
            f = layer(f)
        return f

    def backward(self, b):
        for layer in reversed(self.layers):
            b = layer.backward(b)

    def post_backward(self, p):
        for layer in self.layers:
            layer.post_backward(p)

    def scale_grad(self):
        for layer in self.layers:
            layer.scale_grad()


def classification_loss(z, y):
    return torch.logsumexp(z, dim=0) - torch.sum(z * y)


def get_loss(z, y, loss_fn):
    return torch.func.vmap(loss_fn, in_dims=(0, 0))(z, y)


def get_gradient(z, y, loss_fn):
    return torch.func.vmap(torch.func.grad(loss_fn), in_dims=(0, 0))(z, y)


def get_first_order_term(mlp):
    scaled_nabla = mlp.scaled_nabla()
    nabla = mlp.nabla()
    return sum([torch.tensordot(scaled_nabla_k, nabla_k, dims=2) for scaled_nabla_k, nabla_k in zip(scaled_nabla, nabla)])


def get_second_order_term(x, y, mlp, loss_fn):
    scaled_nabla = mlp.scaled_nabla()
    func = lambda *theta: torch.mean(get_loss(mlp.fwd(theta, x), y, loss_fn))
    vhp = torch.autograd.functional.vhp(func, mlp.theta(), v=scaled_nabla)[1]
    return sum([torch.tensordot(scaled_nabla_k, vhp_k, dims=2) for scaled_nabla_k, vhp_k in zip(scaled_nabla, vhp)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",                                default="/home/logs/")
    parser.add_argument("--data_dir",                               default="/home/data/")
    parser.add_argument("--random_seed_x",              type=int,   default=0)
    parser.add_argument("--random_seed_theta",          type=int,   default=0)
    parser.add_argument("--log_2_m",                    type=int,   default=8)
    parser.add_argument("--l",                          type=int,   default=8)
    parser.add_argument("--log_2_n",                    type=int,   default=8)
    parser.add_argument("--a",                          type=float, default=0.5)
    parser.add_argument("--b",                          type=float, default=0.5)
    parser.add_argument("--q",                          type=float, default=1)
    parser.add_argument("--r",                          type=int,   default=2)
    parser.add_argument("--dataset",                                default="MNIST", choices=[
        "MNIST", "CIFAR10", "CIFAR100"
    ])
    parser.add_argument("--p",                          type=float, default=1.0)
    parser.add_argument("--num_steps",                  type=int,   default=2000)
    parser.add_argument("--lr",                         type=float, default=2.0)

    args = parser.parse_args()
    print(args)

    m = 2 ** args.log_2_m
    l = args.l
    n = 2 ** args.log_2_n

    log_name = f"fl_beyond_eos_{args.dataset}_p_{args.p}_q_{args.q}_r_{args.r}_n_{n}_m_{m}_l_{l}_lr_{args.lr}_a_{args.a}_b_{args.b}_num_steps_{args.num_steps}"
    log_dir = args.log_dir + log_name + "_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
    os.makedirs(log_dir)
    print("Logging to:", log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device("cuda")

    m_0 = {
        "MNIST": 28 * 28,
        "CIFAR10": 32 * 32 * 3,
        "CIFAR100": 32 * 32 * 3
    }[args.dataset]
    m_l = {
        "MNIST": 10,
        "CIFAR10": 10,
        "CIFAR100": 100
    }[args.dataset]

    q = args.q
    r = args.r
    a = args.a
    b = args.b

    lr = args.lr

    loss_fn = classification_loss

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            {
                "MNIST": transforms.Normalize((0.13062754273414612,), (0.30810779333114624,)),
                "CIFAR10": transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                "CIFAR100": transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            }[args.dataset],
            lambda x: torch.flatten(x) * m_0 ** (-1 / 2)
        ]
    )
    target_transform = transforms.Compose(
        [lambda y: torch.LongTensor([y]), lambda y: torch.squeeze(F.one_hot(y, m_l)).float()]
    )
    dataset_fn = {
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100
    }[args.dataset]
    train_data = dataset_fn(
        args.data_dir, train=True, download=True, transform=transform, target_transform=target_transform
    )
    test_data = dataset_fn(
        args.data_dir, train=False, download=True, transform=transform, target_transform=target_transform
    )
    get_train_data = lambda: next(
        iter(torch.utils.data.DataLoader(train_data, batch_size=n, shuffle=True))
    )
    get_test_data = lambda: next(
        iter(torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False))
    )

    torch.manual_seed(args.random_seed_theta)
    numlp = NUMLP(m_0, m_l, m, l, a, b, q, r)
    numlp.to(device=device)
    optimizer = torch.optim.SGD(numlp.parameters(), lr=lr)
    writer.add_scalar("dim_theta", sum([math.prod(layer.theta.size()) for layer in numlp.layers]), global_step=0)
    writer.add_scalar("lr", lr, global_step=l)
    writer.flush()

    torch.manual_seed(args.random_seed_x)
    x, y = get_train_data()
    x, y = x.to(device), y.to(device)
    x_test, y_test = get_test_data()
    x_test, y_test = x_test.to(device), y_test.to(device)
    progressbar = tqdm(range(args.num_steps), desc=f"training...")
    for step in progressbar:
        optimizer.zero_grad()
        with torch.no_grad():
            output = numlp(x)
        gradient = get_gradient(output, y, loss_fn)
        with torch.no_grad():
            numlp.backward(gradient)
            loss = torch.mean(get_loss(output, y, loss_fn))
        writer.add_scalar("train_loss", loss, global_step=step)
        with torch.no_grad():
            numlp.post_backward(args.p)

        first_order_term = get_first_order_term(numlp)
        second_order_term = get_second_order_term(x, y, numlp, loss_fn)
        effective_sharpness = second_order_term / first_order_term
        writer.add_scalar("first_order_term", first_order_term, global_step=step)
        writer.add_scalar("second_order_term", second_order_term, global_step=step)
        writer.add_scalar("effective_sharpness", effective_sharpness, global_step=step)
        writer.add_scalar(
            f"lr_times_effective_sharpness_over_2",
            lr * effective_sharpness / 2,
            global_step=step
        )
        for k in range(l + 1):
            writer.add_scalar(f"grad_scale_{k + 1}", numlp.layers[k].grad_scale, global_step=step)
            f_norm = numlp.layers[k].F_prev_tr ** (1 / 2)
            writer.add_scalar(f"norm_f_{k}", f_norm, global_step=step)
            b_norm = numlp.layers[k].B_tr ** (1 / 2)
            writer.add_scalar(f"norm_b_{k + 1}", b_norm, global_step=step)
            f_rank = numlp.layers[k].F_prev_tr ** 2 / numlp.layers[k].F_prev_fr ** 2
            writer.add_scalar(f"rank_f_{k}", f_rank, global_step=step)
            b_rank = numlp.layers[k].B_tr ** 2 / numlp.layers[k].B_fr ** 2
            writer.add_scalar(f"rank_b_{k + 1}", b_rank, global_step=step)
            f_b_cos = torch.trace(numlp.layers[k].F_prev @ numlp.layers[k].B) / numlp.layers[k].F_prev_fr / numlp.layers[k].B_fr
            writer.add_scalar(f"cos_f_{k}_b_{k + 1}", f_b_cos, global_step=step)
        progressbar.set_description(f"training... loss = {loss:.5f}")
        numlp.scale_grad()
        optimizer.step()
        x, y = get_train_data()
        x, y = x.to(device), y.to(device)

    writer.flush()

    writer.close()


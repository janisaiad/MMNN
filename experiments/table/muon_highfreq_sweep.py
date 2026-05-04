#!/usr/bin/env python3
"""High-frequency MMNN sweeps with SGD, AdamW, Muon, low-rank Muon, and HT-Muon."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.table.mmnn_vs import MMNN


EPS = 1e-12


@dataclass(frozen=True)
class RunConfig:
    model: str
    optimizer: str
    dim: int
    target: str
    freq: int
    depth: int
    width: int
    rank: int
    steps: int
    batch_size: int
    lr: float
    seed: int
    fix_wb: bool
    muon_ns_steps: int
    lowrank_update_rank: int
    ht_alpha: float

    @property
    def name(self) -> str:
        fix = "fixWb" if self.fix_wb else "trainWb"
        return (
            f"{self.model}_{self.optimizer}_{self.target}_d{self.dim}_f{self.freq}_"
            f"r{self.rank}_w{self.width}_L{self.depth}_bs{self.batch_size}_"
            f"steps{self.steps}_{fix}_s{self.seed}"
        )


class MLP(nn.Module):
    """Small ReLU MLP baseline for the same synthetic targets."""

    def __init__(self, dim: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_fn(x: torch.Tensor, target: str, freq: int) -> torch.Tensor:
    if target == "cos1d":
        return torch.cos(2.0 * math.pi * float(freq) * x[:, :1])
    if target == "chirp1d":
        low_freq = max(1.0, float(freq) / 3.0)
        return torch.cos(float(freq) * math.pi * x[:, :1] ** 2) - 0.8 * torch.cos(low_freq * math.pi * x[:, :1] ** 2)
    if target == "sumcos":
        y = torch.zeros_like(x[:, :1])
        for harmonic in range(1, max(1, freq) + 1):
            y = y + torch.cos(2.0 * math.pi * float(harmonic) * x[:, :1])
        return y / math.sqrt(float(max(1, freq)))
    if target == "expcos":
        y = torch.zeros_like(x[:, :1])
        for harmonic in range(0, max(1, freq) + 1):
            y = y + torch.cos((2.0 ** harmonic) * math.pi * x[:, :1])
        return y / math.sqrt(float(max(1, freq) + 1))
    if target == "axis_sum":
        return torch.sum(torch.cos(2.0 * math.pi * float(freq) * x), dim=1, keepdim=True) / math.sqrt(x.shape[1])
    if target == "soft_axis":
        return torch.sum(torch.cos(math.pi * float(freq) * x), dim=1, keepdim=True) / math.sqrt(x.shape[1])
    if target == "radial":
        radius = torch.linalg.norm(x, dim=1, keepdim=True)
        return torch.cos(2.0 * math.pi * float(freq) * radius)
    if target == "pairwise":
        sq = x ** 2
        y = torch.sum(torch.cos(2.0 * math.pi * float(freq) * sq), dim=1, keepdim=True) / math.sqrt(x.shape[1])
        if x.shape[1] >= 2:
            y = y + 0.35 * torch.cos(2.0 * math.pi * float(freq) * torch.prod(x[:, :2], dim=1, keepdim=True))
        return y
    raise ValueError(f"unknown target: {target}")


def sample_batch(batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
    return 2.0 * torch.rand(batch_size, dim, device=device) - 1.0


def build_model(config: RunConfig, device: torch.device) -> nn.Module:
    if config.model == "mmnn":
        ranks = [config.dim] + [config.rank] * config.depth + [1]
        widths = [config.width] * (config.depth + 1)
        return MMNN(ranks=ranks, widths=widths, device=str(device), ResNet=False, fixWb=config.fix_wb).to(device)
    if config.model == "mlp":
        return MLP(config.dim, config.width, config.depth).to(device)
    raise ValueError(f"unknown model: {config.model}")


def polar_newton_schulz(x: torch.Tensor, steps: int) -> torch.Tensor:
    if x.ndim != 2:
        return x
    if torch.linalg.matrix_norm(x).item() <= EPS:
        return torch.zeros_like(x)
    transposed = x.shape[0] > x.shape[1]
    y = x.mT if transposed else x
    y = y / (torch.linalg.matrix_norm(y) + EPS)
    for _ in range(steps):
        a = y @ y.mT
        y = 3.4445 * y + (-4.7750) * (a @ y) + 2.0315 * (a @ a @ y)
    return y.mT if transposed else y


def truncated_polar(x: torch.Tensor, update_rank: int) -> torch.Tensor:
    if x.ndim != 2:
        return x
    if torch.linalg.matrix_norm(x).item() <= EPS:
        return torch.zeros_like(x)
    max_rank = min(x.shape)
    rank = max(1, min(update_rank, max_rank))
    u, _s, vh = torch.linalg.svd(x, full_matrices=False)
    return u[:, :rank] @ vh[:rank, :]


def powerlaw_polar(x: torch.Tensor, alpha: float, update_rank: int | None = None) -> torch.Tensor:
    if x.ndim != 2:
        return x
    if torch.linalg.matrix_norm(x).item() <= EPS:
        return torch.zeros_like(x)
    u, _s, vh = torch.linalg.svd(x, full_matrices=False)
    max_rank = min(x.shape)
    rank = max_rank if update_rank is None else max(1, min(update_rank, max_rank))
    weights = torch.arange(1, rank + 1, device=x.device, dtype=x.dtype).pow(-alpha)
    weights = weights / (weights[0] + EPS)
    return (u[:, :rank] * weights.unsqueeze(0)) @ vh[:rank, :]


class Muon(torch.optim.Optimizer):
    """Muon-style optimizer for matrix parameters, with AdamW fallback for vectors."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        fallback_lr: float | None = None,
        lowrank_update_rank: int | None = None,
        ht_alpha: float | None = None,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "fallback_lr": lr if fallback_lr is None else fallback_lr,
            "lowrank_update_rank": lowrank_update_rank,
            "ht_alpha": ht_alpha,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            fallback_lr = group["fallback_lr"]
            lowrank_update_rank = group["lowrank_update_rank"]
            ht_alpha = group["ht_alpha"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(beta).add_(grad, alpha=1.0 - beta)
                if param.ndim == 2:
                    if ht_alpha is not None:
                        update = powerlaw_polar(momentum_buffer, ht_alpha, lowrank_update_rank)
                    elif lowrank_update_rank is None:
                        update = polar_newton_schulz(momentum_buffer, ns_steps)
                    else:
                        update = truncated_polar(momentum_buffer, lowrank_update_rank)
                    param.add_(update, alpha=-lr)
                else:
                    param.add_(momentum_buffer, alpha=-fallback_lr)
        return loss


def make_optimizer(config: RunConfig, model: nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.lr, momentum=0.9)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=1e-5)
    if config.optimizer == "muon":
        return Muon(params, lr=config.lr, momentum=0.95, ns_steps=config.muon_ns_steps, weight_decay=1e-5)
    if config.optimizer == "lowrank_muon":
        return Muon(
            params,
            lr=config.lr,
            momentum=0.95,
            ns_steps=config.muon_ns_steps,
            weight_decay=1e-5,
            lowrank_update_rank=config.lowrank_update_rank,
        )
    if config.optimizer == "ht_muon":
        return Muon(
            params,
            lr=config.lr,
            momentum=0.95,
            ns_steps=config.muon_ns_steps,
            weight_decay=1e-5,
            lowrank_update_rank=config.lowrank_update_rank,
            ht_alpha=config.ht_alpha,
        )
    raise ValueError(f"unknown optimizer: {config.optimizer}")


@torch.no_grad()
def evaluate(model: nn.Module, config: RunConfig, device: torch.device, n_test: int) -> dict[str, float]:
    model.eval()
    losses = []
    max_errors = []
    remaining = n_test
    while remaining > 0:
        current = min(8192, remaining)
        x = sample_batch(current, config.dim, device)
        y = target_fn(x, config.target, config.freq)
        pred = model(x)
        err = pred - y
        losses.append(torch.mean(err ** 2).item())
        max_errors.append(torch.max(torch.abs(err)).item())
        remaining -= current
    return {"test_mse": float(np.mean(losses)), "test_max_error": float(np.max(max_errors))}


def spectral_stats(model: nn.Module) -> dict[str, float]:
    ranks = []
    top_singulars = []
    slopes = []
    for param in model.parameters():
        if param.requires_grad and param.ndim == 2 and param.grad is not None:
            singulars = torch.linalg.svdvals(param.grad.detach().float()).cpu().numpy()
            if singulars.size == 0:
                continue
            energy = singulars ** 2
            ranks.append(float((energy.sum() ** 2) / (np.sum(energy ** 2) + EPS)))
            top_singulars.append(float(singulars[0]))
            if singulars.size >= 3 and np.all(singulars[: min(8, singulars.size)] > 0):
                count = min(8, singulars.size)
                xs = np.log(np.arange(1, count + 1, dtype=float))
                ys = np.log(singulars[:count] + EPS)
                slopes.append(float(np.polyfit(xs, ys, 1)[0]))
    if not ranks:
        return {"grad_effective_rank": float("nan"), "grad_top_singular": float("nan"), "grad_spectral_slope": float("nan")}
    return {
        "grad_effective_rank": float(np.mean(ranks)),
        "grad_top_singular": float(np.mean(top_singulars)),
        "grad_spectral_slope": float(np.mean(slopes)) if slopes else float("nan"),
    }


@torch.no_grad()
def weight_spectral_stats(model: nn.Module) -> dict[str, float]:
    ranks = []
    slopes = []
    for param in model.parameters():
        if param.requires_grad and param.ndim == 2:
            singulars = torch.linalg.svdvals(param.detach().float()).cpu().numpy()
            if singulars.size == 0:
                continue
            energy = singulars ** 2
            ranks.append(float((energy.sum() ** 2) / (np.sum(energy ** 2) + EPS)))
            if singulars.size >= 3 and np.all(singulars[: min(8, singulars.size)] > 0):
                count = min(8, singulars.size)
                xs = np.log(np.arange(1, count + 1, dtype=float))
                ys = np.log(singulars[:count] + EPS)
                slopes.append(float(np.polyfit(xs, ys, 1)[0]))
    if not ranks:
        return {"weight_effective_rank": float("nan"), "weight_spectral_slope": float("nan")}
    return {
        "weight_effective_rank": float(np.mean(ranks)),
        "weight_spectral_slope": float(np.mean(slopes)) if slopes else float("nan"),
    }


def save_loss_plot(run_dir: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    steps = [row["step"] for row in history]
    train = [row["train_mse"] for row in history]
    test = [row["test_mse"] for row in history]
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(steps, train, label="train batch MSE")
    plt.plot(steps, test, label="test MSE")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.title("High-frequency sweep")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=220)
    plt.close()


def train_one(config: RunConfig, out_dir: Path, device: torch.device, overwrite: bool) -> dict[str, object]:
    run_dir = out_dir / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        with open(metrics_path) as f:
            return json.load(f)

    set_seed(config.seed)
    model = build_model(config, device)
    optimizer = make_optimizer(config, model)
    nparams = sum(p.numel() for p in model.parameters())
    ntrainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eval_points = max(4096, 64 * config.freq * max(1, config.dim))
    log_every = max(1, min(250, config.steps // 20))
    history: list[dict[str, float]] = []
    start_time = time.time()
    latest_train = float("nan")
    latest_spectral = {"grad_effective_rank": float("nan"), "grad_top_singular": float("nan")}

    for step in range(1, config.steps + 1):
        model.train()
        x = sample_batch(config.batch_size, config.dim, device)
        y = target_fn(x, config.target, config.freq)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        latest_spectral = spectral_stats(model)
        optimizer.step()
        latest_train = float(loss.detach().cpu().item())

        if step == 1 or step % log_every == 0 or step == config.steps:
            eval_metrics = evaluate(model, config, device, eval_points)
            row = {
                "step": float(step),
                "train_mse": latest_train,
                **eval_metrics,
                **latest_spectral,
                "lr": config.lr,
            }
            history.append(row)
            print(
                f"{config.name} step {step}/{config.steps} "
                f"train={latest_train:.3e} test={eval_metrics['test_mse']:.3e} "
                f"erank={latest_spectral['grad_effective_rank']:.2f}",
                flush=True,
            )

    final_eval = evaluate(model, config, device, eval_points)
    final_weight_stats = weight_spectral_stats(model)
    elapsed = time.time() - start_time
    row = {
        **asdict(config),
        "name": config.name,
        "nparams": nparams,
        "ntrainable": ntrainable,
        "elapsed_sec": elapsed,
        "final_train_mse": latest_train,
        **final_eval,
        **latest_spectral,
        **final_weight_stats,
        "eval_points": eval_points,
    }
    with open(metrics_path, "w") as f:
        json.dump(row, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    torch.save(model.state_dict(), run_dir / "model_parameters.pth")
    save_loss_plot(run_dir, history)
    return row


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def parse_bools(value: str) -> list[bool]:
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    return [mapping[item.strip().lower()] for item in value.split(",") if item.strip()]


def build_configs(args: argparse.Namespace) -> list[RunConfig]:
    configs: list[RunConfig] = []
    for seed in args.seeds:
        for model in args.models:
            for optimizer in args.optimizers:
                for dim in args.dims:
                    for target in args.targets:
                        for freq in args.freqs:
                            for depth in args.depths:
                                for rank in args.ranks:
                                    if model == "mlp" and rank != args.ranks[0]:
                                        continue
                                    for fix_wb in args.fix_wb_modes:
                                        lr = args.muon_lr if optimizer in {"muon", "lowrank_muon", "ht_muon"} else args.lr
                                        configs.append(
                                            RunConfig(
                                                model=model,
                                                optimizer=optimizer,
                                                dim=dim,
                                                target=target,
                                                freq=freq,
                                                depth=depth,
                                                width=args.width,
                                                rank=rank,
                                                steps=args.steps,
                                                batch_size=args.batch_size,
                                                lr=lr,
                                                seed=seed,
                                                fix_wb=fix_wb,
                                                muon_ns_steps=args.muon_ns_steps,
                                                lowrank_update_rank=args.lowrank_update_rank,
                                                ht_alpha=args.ht_alpha,
                                            )
                                        )
    return configs[: args.max_runs] if args.max_runs is not None else configs


def write_summary(out_dir: Path, rows: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    fields = [
        "name",
        "model",
        "optimizer",
        "dim",
        "target",
        "freq",
        "depth",
        "width",
        "rank",
        "steps",
        "batch_size",
        "lr",
        "seed",
        "fix_wb",
        "nparams",
        "ntrainable",
        "final_train_mse",
        "test_mse",
        "test_max_error",
        "grad_effective_rank",
        "grad_top_singular",
        "grad_spectral_slope",
        "weight_effective_rank",
        "weight_spectral_slope",
        "elapsed_sec",
    ]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="High-frequency Muon/MMNN sweep")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/table/results_muon_highfreq"))
    parser.add_argument("--models", type=parse_strings, default=["mmnn"])
    parser.add_argument("--optimizers", type=parse_strings, default=["adamw", "muon", "lowrank_muon"])
    parser.add_argument("--dims", type=parse_ints, default=[1, 2, 3])
    parser.add_argument("--targets", type=parse_strings, default=["cos1d", "axis_sum", "radial"])
    parser.add_argument("--freqs", type=parse_ints, default=[8, 16, 32, 64])
    parser.add_argument("--ranks", type=parse_ints, default=[4, 8, 16])
    parser.add_argument("--seeds", type=parse_ints, default=[0])
    parser.add_argument("--depths", type=parse_ints, default=[3])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--muon-lr", type=float, default=1e-2)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--lowrank-update-rank", type=int, default=4)
    parser.add_argument("--ht-alpha", type=float, default=0.75)
    parser.add_argument("--fix-wb-modes", type=parse_bools, default=[True])
    parser.add_argument("--no-fix-wb", action="store_true", help="Deprecated shorthand equivalent to --fix-wb-modes false")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.no_fix_wb:
        args.fix_wb_modes = [False]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = build_configs(args)
    print(f"device={device} runs={len(configs)} out={args.out_dir}", flush=True)
    rows: list[dict[str, object]] = []
    existing_summary = args.out_dir / "summary.json"
    if existing_summary.exists() and not args.overwrite:
        with open(existing_summary) as f:
            rows = json.load(f)
    by_name = {str(row["name"]): row for row in rows}
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.name}", flush=True)
        row = train_one(config, args.out_dir, device, args.overwrite)
        by_name[str(row["name"])] = row
        write_summary(args.out_dir, list(by_name.values()))
    print(f"saved {args.out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

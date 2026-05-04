#!/usr/bin/env python3
"""Batch-size-1 multidimensional symmetry experiments for MMNNs and MLPs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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

from experiments.table.mmnn_vs import MMNN  # noqa: E402


OUT_ROOT = Path(__file__).resolve().parent / "results" / "multidim_batch1_symmetry"
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    dim: int
    target: str
    model: str
    depth: int
    width: int
    rank: int
    freq: int
    n_train: int
    batch_size: int
    epochs: int
    lr: float
    seed: int
    train_distribution: str
    optimizer: str

    @property
    def name(self) -> str:
        return (
            f"{self.model}_d{self.dim}_{self.target}_f{self.freq}_"
            f"r{self.rank}_w{self.width}_L{self.depth}_N{self.n_train}_bs{self.batch_size}_"
            f"{self.train_distribution}_{self.optimizer}_s{self.seed}"
        )


class MLP(nn.Module):
    def __init__(self, dim: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = dim
        for _ in range(depth):
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_fn(x: torch.Tensor, target: str, freq: int) -> torch.Tensor:
    if target == "gaussian":
        radius_sq = torch.sum(x ** 2, dim=1, keepdim=True)
        return torch.exp(-float(freq) * radius_sq)
    if target == "quadratic":
        radius_sq = torch.sum(x ** 2, dim=1, keepdim=True)
        return 1.0 - 0.5 * radius_sq
    if target == "soft_axis":
        return torch.sum(torch.cos(math.pi * freq * x), dim=1, keepdim=True) / math.sqrt(x.shape[1])
    if target == "radial":
        radius = torch.linalg.norm(x, dim=1, keepdim=True)
        return torch.cos(2.0 * math.pi * freq * radius)
    if target == "axis_sum":
        return torch.sum(torch.cos(2.0 * math.pi * freq * x), dim=1, keepdim=True) / math.sqrt(x.shape[1])
    if target == "pairwise":
        sq = x ** 2
        base = torch.sum(torch.cos(2.0 * math.pi * freq * sq), dim=1, keepdim=True) / math.sqrt(x.shape[1])
        if x.shape[1] >= 2:
            base = base + 0.35 * torch.cos(2.0 * math.pi * freq * torch.prod(x[:, :2], dim=1, keepdim=True))
        return base
    raise ValueError(f"unknown target: {target}")


def remove_near_mirrors(points: np.ndarray, min_distance: float) -> np.ndarray:
    kept: list[np.ndarray] = []
    for point in points:
        if not kept:
            kept.append(point)
            continue
        current = np.stack(kept, axis=0)
        nearest_mirror = float(np.min(np.linalg.norm(current + point[None, :], axis=1)))
        if nearest_mirror >= min_distance:
            kept.append(point)
    return np.stack(kept, axis=0)


def make_asymmetric_train_points(n: int, dim: int, seed: int, distribution: str) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    if distribution == "unpaired_uniform":
        candidates = rng.uniform(-1.0, 1.0, size=(max(20 * n, 2048), dim)).astype(np.float32)
        points = remove_near_mirrors(candidates, min_distance=0.06)
        if points.shape[0] < n:
            points = candidates
        return torch.from_numpy(points[:n].astype(np.float32))
    if distribution == "positive_bias":
        mixture = rng.uniform(-1.0, 1.0, size=(n, dim)).astype(np.float32)
        mask = rng.random(size=(n, dim)) < 0.72
        positive_bias = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
        points = np.where(mask, positive_bias, mixture).astype(np.float32)
        return torch.from_numpy(points)
    mixture = rng.uniform(-1.0, 1.0, size=(n, dim)).astype(np.float32)
    return torch.from_numpy(mixture)


def asymmetry_score(x: torch.Tensor) -> float:
    x_np = x.detach().cpu().numpy()
    sample = x_np[: min(512, x_np.shape[0])]
    distances = []
    for point in sample:
        mirror = -point
        distances.append(float(np.min(np.linalg.norm(x_np - mirror[None, :], axis=1))))
    return float(np.mean(distances))


def build_model(config: Config, device: torch.device) -> nn.Module:
    if config.model == "mmnn":
        ranks = [config.dim] + [config.rank] * config.depth + [1]
        widths = [config.width] * (config.depth + 1)
        return MMNN(ranks=ranks, widths=widths, device=str(device), ResNet=False, fixWb=True)
    if config.model == "mlp":
        return MLP(config.dim, config.width, config.depth).to(device)
    raise ValueError(f"unknown model: {config.model}")


def collect_partials(model: nn.Module, x: torch.Tensor, model_name: str) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    z = x
    if model_name == "mmnn":
        depth = int(getattr(model, "depth"))
        for j in range(depth - 1):
            z = model.fcs[2 * j](z)
            z = torch.relu(z)
            z = model.fcs[2 * j + 1](z)
            values.append(z)
        return values
    if model_name == "mlp":
        for layer in model.net:
            z = layer(z)
            if isinstance(layer, nn.ReLU):
                values.append(z)
        return values
    raise ValueError(f"unknown model: {model_name}")


def transform_batch(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "neg":
        return -x
    if name == "flip_first":
        y = x.clone()
        y[:, 0] = -y[:, 0]
        return y
    if name == "reverse":
        return torch.flip(x, dims=[1])
    if name == "rot90_2d":
        if x.shape[1] != 2:
            return x
        return torch.stack([-x[:, 1], x[:, 0]], dim=1)
    return x


def relative_defect(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((torch.mean((a - b) ** 2) / (torch.mean(a ** 2) + EPS)).detach().cpu().item())


def active_partial_defect(model: nn.Module, x: torch.Tensor, transform: str, model_name: str) -> tuple[float, float]:
    with torch.no_grad():
        p_a = collect_partials(model, x, model_name)
        p_b = collect_partials(model, transform_batch(x, transform), model_name)
    if not p_a:
        return float("nan"), float("nan")
    all_means = []
    last_mean = float("nan")
    for idx, (a, b) in enumerate(zip(p_a, p_b)):
        energy = torch.mean(a ** 2, dim=0)
        defect = torch.mean((a - b) ** 2, dim=0) / (energy + EPS)
        threshold = torch.quantile(energy.detach(), 0.5)
        mask = energy >= torch.maximum(threshold, torch.tensor(1e-8, device=energy.device))
        active = defect[mask] if torch.any(mask) else defect
        mean_value = float(torch.mean(active).detach().cpu().item())
        all_means.append(mean_value)
        if idx == len(p_a) - 1:
            last_mean = mean_value
    return last_mean, float(np.mean(all_means))


def first_layer_mirror_metrics(model: nn.Module, model_name: str) -> dict[str, float]:
    if model_name != "mmnn":
        return {
            "mirror_distance_median": float("nan"),
            "mirror_mismatch_median": float("nan"),
            "mirror_corr_median": float("nan"),
        }
    first = model.fcs[0]
    second = model.fcs[1]
    weights = first.weight.detach().cpu().numpy().astype(np.float64)
    biases = first.bias.detach().cpu().numpy().astype(np.float64)
    outgoing = second.weight.detach().cpu().numpy().astype(np.float64)
    distances = []
    mismatches = []
    correlations = []
    for j in range(weights.shape[0]):
        squared = np.sum((weights + weights[j][None, :]) ** 2, axis=1) + (biases - biases[j]) ** 2
        squared[j] = np.inf
        k = int(np.argmin(squared))
        cj = outgoing[:, j]
        ck = outgoing[:, k]
        distances.append(float(np.sqrt(squared[k])))
        mismatches.append(float(np.mean((cj - ck) ** 2) / (np.mean(cj ** 2 + ck ** 2) + EPS)))
        correlations.append(float(np.dot(cj, ck) / (np.linalg.norm(cj) * np.linalg.norm(ck) + EPS)))
    return {
        "mirror_distance_median": float(np.median(distances)),
        "mirror_mismatch_median": float(np.median(mismatches)),
        "mirror_corr_median": float(np.median(correlations)),
    }


def expansion_layer_mirror_metrics(model: nn.Module, model_name: str) -> dict[str, float]:
    if model_name != "mmnn":
        return {
            "all_layer_mirror_distance_median": float("nan"),
            "all_layer_mirror_mismatch_median": float("nan"),
            "all_layer_mirror_corr_median": float("nan"),
            "last_layer_mirror_distance_median": float("nan"),
            "last_layer_mirror_mismatch_median": float("nan"),
            "last_layer_mirror_corr_median": float("nan"),
        }
    per_layer_distance = []
    per_layer_mismatch = []
    per_layer_corr = []
    depth = int(getattr(model, "depth"))
    for layer_index in range(depth):
        expansion = model.fcs[2 * layer_index]
        contraction = model.fcs[2 * layer_index + 1]
        weights = expansion.weight.detach().cpu().numpy().astype(np.float64)
        biases = expansion.bias.detach().cpu().numpy().astype(np.float64)
        outgoing = contraction.weight.detach().cpu().numpy().astype(np.float64)
        distances = []
        mismatches = []
        correlations = []
        for j in range(weights.shape[0]):
            squared = np.sum((weights + weights[j][None, :]) ** 2, axis=1) + (biases - biases[j]) ** 2
            squared[j] = np.inf
            k = int(np.argmin(squared))
            cj = outgoing[:, j]
            ck = outgoing[:, k]
            distances.append(float(np.sqrt(squared[k])))
            mismatches.append(float(np.mean((cj - ck) ** 2) / (np.mean(cj ** 2 + ck ** 2) + EPS)))
            correlations.append(float(np.dot(cj, ck) / (np.linalg.norm(cj) * np.linalg.norm(ck) + EPS)))
        per_layer_distance.append(float(np.median(distances)))
        per_layer_mismatch.append(float(np.median(mismatches)))
        per_layer_corr.append(float(np.median(correlations)))
    return {
        "all_layer_mirror_distance_median": float(np.mean(per_layer_distance)),
        "all_layer_mirror_mismatch_median": float(np.mean(per_layer_mismatch)),
        "all_layer_mirror_corr_median": float(np.mean(per_layer_corr)),
        "last_layer_mirror_distance_median": float(per_layer_distance[-1]),
        "last_layer_mirror_mismatch_median": float(per_layer_mismatch[-1]),
        "last_layer_mirror_corr_median": float(per_layer_corr[-1]),
    }


def evaluate(model: nn.Module, config: Config, train_x: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> dict[str, float]:
    model.eval()
    transforms = ["neg", "flip_first", "reverse"]
    if config.dim == 2:
        transforms.append("rot90_2d")
    with torch.no_grad():
        pred = model(test_x)
        train_y = target_fn(train_x, config.target, config.freq)
        train_pred = model(train_x)
        test_mse = float(torch.mean((pred - test_y) ** 2).detach().cpu().item())
        train_mse = float(torch.mean((train_pred - train_y) ** 2).detach().cpu().item())
    metrics: dict[str, float] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_asymmetry_score": asymmetry_score(train_x),
    }
    for transform in transforms:
        with torch.no_grad():
            a = model(test_x)
            b = model(transform_batch(test_x, transform))
        metrics[f"output_{transform}_defect"] = relative_defect(a, b)
        last, mean = active_partial_defect(model, test_x, transform, config.model)
        metrics[f"last_partial_{transform}_defect"] = last
        metrics[f"mean_partial_{transform}_defect"] = mean
    metrics.update(first_layer_mirror_metrics(model, config.model))
    metrics.update(expansion_layer_mirror_metrics(model, config.model))
    return metrics


def save_run_plots(run_dir: Path, losses: list[float], metrics: dict[str, float], config: Config) -> None:
    plt.figure(figsize=(6.4, 4.0))
    plt.semilogy(np.arange(1, len(losses) + 1), losses, color="#1b4e8a", linewidth=1.8)
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title(config.name)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=260)
    plt.close()
    labels = [key.replace("_defect", "").replace("_", "\n") for key in metrics if key.startswith("output_")]
    out_vals = [metrics[key] for key in metrics if key.startswith("output_")]
    part_vals = [metrics[key.replace("output_", "last_partial_")] for key in metrics if key.startswith("output_")]
    x = np.arange(len(labels))
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(x - 0.18, np.asarray(out_vals) + EPS, width=0.36, label="output", color="#2a9d8f")
    plt.bar(x + 0.18, np.asarray(part_vals) + EPS, width=0.36, label="last partial", color="#e76f51")
    plt.yscale("log")
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("relative defect")
    plt.title("Symmetry defects")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(run_dir / "symmetry_defects.png", dpi=260)
    plt.close()


def train_one(config: Config, overwrite: bool) -> dict[str, object]:
    run_dir = OUT_ROOT / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "metrics.json"
    if summary_path.exists() and not overwrite:
        with open(summary_path) as f:
            return json.load(f)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x_cpu = make_asymmetric_train_points(config.n_train, config.dim, config.seed, config.train_distribution)
    train_y_cpu = target_fn(train_x_cpu, config.target, config.freq)
    test_x = torch.from_numpy(np.random.default_rng(1000 + config.seed).uniform(-1.0, 1.0, size=(4096, config.dim)).astype(np.float32)).to(device)
    test_y = target_fn(test_x, config.target, config.freq)
    train_x_gpu = train_x_cpu.to(device)
    train_y_gpu = train_y_cpu.to(device)
    model = build_model(config, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=config.lr)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=1e-5)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=config.lr, momentum=0.0)
    else:
        raise ValueError(f"unknown optimizer: {config.optimizer}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1), eta_min=config.lr * 0.03)
    losses: list[float] = []
    progress_path = run_dir / "progress.json"
    best_loss = float("inf")
    best_path = run_dir / "model_parameters_best.pth"
    for epoch in range(config.epochs):
        model.train()
        epoch_losses = []
        permutation = torch.randperm(train_x_gpu.shape[0], device=device)
        for start in range(0, train_x_gpu.shape[0], config.batch_size):
            indices = permutation[start: start + config.batch_size]
            xb = train_x_gpu[indices]
            yb = train_y_gpu[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        scheduler.step()
        losses.append(float(np.mean(epoch_losses)))
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            torch.save(model.state_dict(), best_path)
        if (epoch + 1) % 200 == 0:
            print(f"  {config.name} epoch {epoch + 1}/{config.epochs} loss={losses[-1]:.4e}", flush=True)
        if (epoch + 1) % 1000 == 0:
            with open(progress_path, "w") as f:
                json.dump({
                    "name": config.name,
                    "epoch": epoch + 1,
                    "latest_train_loss": losses[-1],
                    "best_train_loss": float(np.min(losses)),
                }, f, indent=2)
            torch.save(model.state_dict(), run_dir / "model_parameters_progress.pth")
    metrics = evaluate(model, config, train_x_gpu, test_x, test_y)
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        best_metrics = evaluate(model, config, train_x_gpu, test_x, test_y)
        metrics.update({
            f"best_{key}": value for key, value in best_metrics.items()
            if key in {"train_mse", "test_mse", "output_neg_defect", "last_partial_neg_defect", "mirror_mismatch_median", "mirror_corr_median"}
        })
        metrics["best_epoch_loss"] = best_loss
    row: dict[str, object] = {**asdict(config), "name": config.name, **metrics}
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(row, f, indent=2)
    torch.save(model.state_dict(), run_dir / "model_parameters.pth")
    save_run_plots(run_dir, losses, metrics, config)
    return row


def write_summary(rows: list[dict[str, object]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "dim", "target", "model", "depth", "width", "rank", "freq", "n_train", "batch_size", "epochs", "lr", "optimizer", "seed",
        "train_distribution", "train_mse", "test_mse", "best_train_mse", "best_test_mse", "best_epoch_loss",
        "train_asymmetry_score", "output_neg_defect", "best_output_neg_defect", "last_partial_neg_defect", "best_last_partial_neg_defect", "output_flip_first_defect",
        "last_partial_flip_first_defect", "output_reverse_defect", "last_partial_reverse_defect", "output_rot90_2d_defect",
        "last_partial_rot90_2d_defect", "mirror_distance_median", "mirror_mismatch_median", "mirror_corr_median",
        "all_layer_mirror_distance_median", "all_layer_mirror_mismatch_median", "all_layer_mirror_corr_median",
        "last_layer_mirror_distance_median", "last_layer_mirror_mismatch_median", "last_layer_mirror_corr_median",
    ]
    with open(OUT_ROOT / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    plot_summary(rows)


def plot_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    })
    markers = {"mmnn": "o", "mlp": "s"}
    colors = {2: "#1b9e77", 3: "#d95f02"}
    plt.figure(figsize=(7.5, 5.0))
    for row in rows:
        if not np.isfinite(float(row["test_mse"])):
            continue
        plt.scatter(
            float(row["test_mse"]),
            float(row["last_partial_neg_defect"]) + EPS,
            marker=markers[str(row["model"])],
            color=colors[int(row["dim"])],
            s=62,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("test MSE")
    plt.ylabel("last partial defect under $x\\mapsto -x$")
    plt.title("Batch-size-1 asymmetric training: 2D/3D symmetry")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="black", label="MMNN"),
        plt.Line2D([], [], marker="s", linestyle="", color="black", label="MLP"),
        plt.Line2D([], [], marker="o", linestyle="", color=colors[2], label="2D"),
        plt.Line2D([], [], marker="o", linestyle="", color=colors[3], label="3D"),
    ]
    plt.legend(handles=handles, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "multidim_loss_vs_partial_symmetry.png", dpi=280)
    plt.close()
    metrics = ["output_neg_defect", "last_partial_neg_defect", "output_flip_first_defect", "last_partial_flip_first_defect"]
    labels = ["output\nneg", "partial\nneg", "output\nflip", "partial\nflip"]
    groups = [("mmnn", 2), ("mmnn", 3), ("mlp", 2), ("mlp", 3)]
    values = []
    for model, dim in groups:
        subset = [row for row in rows if row["model"] == model and int(row["dim"]) == dim and float(row["test_mse"]) <= 5e-3]
        values.append([np.median([float(row[m]) for row in subset]) if subset else np.nan for m in metrics])
    plt.figure(figsize=(7.6, 4.6))
    x = np.arange(len(metrics))
    width = 0.18
    for idx, ((model, dim), vals) in enumerate(zip(groups, values)):
        plt.bar(x + (idx - 1.5) * width, np.asarray(vals) + EPS, width=width, label=f"{model.upper()} {dim}D")
    plt.yscale("log")
    plt.xticks(x, labels)
    plt.ylabel("median defect among low-loss runs")
    plt.title("Symmetry from fully asymmetric batch-size-1 training")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "multidim_low_loss_defect_bars.png", dpi=280)
    plt.close()
    mmnn_rows = [row for row in rows if row["model"] == "mmnn" and np.isfinite(float(row.get("mirror_mismatch_median", np.nan)))]
    if mmnn_rows:
        plt.figure(figsize=(7.4, 5.0))
        for row in mmnn_rows:
            plt.scatter(
                float(row["mirror_mismatch_median"]) + EPS,
                float(row["last_partial_neg_defect"]) + EPS,
                color=colors[int(row["dim"])],
                s=62,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.5,
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("median first-layer mirror mismatch")
        plt.ylabel("last partial defect under $x\\mapsto -x$")
        plt.title("Weight-space mirror mismatch vs partial symmetry")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "multidim_weightspace_vs_partial.png", dpi=280)
        plt.close()


def config_grid(args: argparse.Namespace) -> list[Config]:
    configs = []
    for seed in args.seeds:
        for dim in args.dims:
            for target in args.targets:
                for model in args.models:
                    for depth in args.depths:
                        rank_values = args.ranks if model == "mmnn" else [0]
                        for rank in rank_values:
                            for freq in args.freqs:
                                configs.append(Config(
                                    dim=dim,
                                    target=target,
                                    model=model,
                                    depth=depth,
                                    width=args.width,
                                    rank=rank,
                                    freq=freq,
                                    n_train=args.n_train,
                                    batch_size=args.batch_size,
                                    epochs=args.epochs,
                                    lr=args.lr,
                                    seed=seed,
                                    train_distribution=args.train_distribution,
                                    optimizer=args.optimizer,
                                ))
    return configs


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def parse_str_list(value: str) -> list[str]:
    return [x for x in value.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=parse_int_list, default=[2, 3])
    parser.add_argument("--targets", type=parse_str_list, default=["radial", "axis_sum", "pairwise"])
    parser.add_argument("--models", type=parse_str_list, default=["mmnn", "mlp"])
    parser.add_argument("--depths", type=parse_int_list, default=[2, 3, 4])
    parser.add_argument("--ranks", type=parse_int_list, default=[8, 16])
    parser.add_argument("--freqs", type=parse_int_list, default=[2, 4])
    parser.add_argument("--seeds", type=parse_int_list, default=[42, 43])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--train-distribution", type=str, default="unpaired_uniform", choices=["unpaired_uniform", "positive_bias", "uniform"])
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    configs = config_grid(args)
    if args.max_runs is not None:
        configs = configs[: args.max_runs]
    print(f"selected {len(configs)} multidimensional configs -> {OUT_ROOT}", flush=True)
    planned_names = {config.name for config in configs}
    rows: list[dict[str, object]] = []
    existing = sorted(OUT_ROOT.glob("*/metrics.json")) if OUT_ROOT.exists() else []
    for path in existing:
        with open(path) as f:
            row = json.load(f)
        if str(row.get("name", "")) in planned_names:
            rows.append(row)
    by_name = {str(row["name"]): row for row in rows}
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.name}", flush=True)
        row = train_one(config, args.overwrite)
        by_name[str(row["name"])] = row
        write_summary(list(by_name.values()))
    write_summary(list(by_name.values()))
    print(f"done -> {OUT_ROOT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

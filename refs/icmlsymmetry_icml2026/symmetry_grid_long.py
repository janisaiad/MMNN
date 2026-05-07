#!/usr/bin/env python3
"""Long symmetry grid for function-space partials and first-layer weight-space mirrors."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.table.mmnn_vs import MMNN  # noqa: E402


OUT_ROOT = Path(__file__).resolve().parent / "results" / "symmetry_grid_long"
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    model_kind: str
    seed: int
    width: int
    rank: int
    depth: int
    n_train: int
    batch_size: int
    epochs: int
    lr: float
    factor: int
    momentum: float


class FullRankMLP(nn.Module):
    def __init__(self, width: int, depth: int, device: torch.device) -> None:
        super().__init__()
        layers: list[nn.Linear] = []
        in_dim = 1
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width, device=device))
            in_dim = width
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(width, 1, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.out(x)

    def partials(self, x: torch.Tensor) -> list[torch.Tensor]:
        values: list[torch.Tensor] = []
        for layer in self.hidden:
            x = torch.relu(layer(x))
            values.append(x)
        return values


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def target_sumcos(x: np.ndarray, factor: int) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    for k in range(1, factor + 1):
        y += np.cos(2.0 * np.pi * k * x)
    return y


def run_name(cfg: Config) -> str:
    return "_".join([
        cfg.model_kind,
        f"seed{cfg.seed}",
        f"W{cfg.width}",
        f"r{cfg.rank}",
        f"L{cfg.depth}",
        f"N{cfg.n_train}",
        f"bs{cfg.batch_size}",
        f"ep{cfg.epochs}",
    ])


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    if cfg.model_kind == "mmnn":
        ranks = [1] + [cfg.rank] * cfg.depth + [1]
        widths = [cfg.width] * (cfg.depth + 1)
        return MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)
    if cfg.model_kind == "mlp":
        return FullRankMLP(width=cfg.width, depth=cfg.depth, device=device)
    raise ValueError(f"unknown model_kind={cfg.model_kind}")


def collect_partials(model: nn.Module, model_kind: str, x: torch.Tensor) -> list[torch.Tensor]:
    if model_kind == "mlp":
        assert isinstance(model, FullRankMLP)
        return model.partials(x)
    z = x
    values: list[torch.Tensor] = []
    depth = int(getattr(model, "depth"))
    for j in range(depth - 1):
        z = model.fcs[2 * j](z)
        z = torch.relu(z)
        z = model.fcs[2 * j + 1](z)
        values.append(z)
    return values


def relative_defect(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    numerator = (a - b) ** 2 if mode == "even" else (a + b) ** 2
    denominator = torch.mean(a ** 2, dim=0) + EPS
    return torch.mean(numerator, dim=0) / denominator


def strict_positive_minima(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 3:
        return np.zeros(values.shape[1], dtype=np.float64)
    left = values[1:-1] < values[:-2]
    right = values[1:-1] < values[2:]
    positive = values[1:-1] > 1e-4
    return np.sum(left & right & positive, axis=0).astype(np.float64)


def function_space_metrics(model: nn.Module, model_kind: str, device: torch.device, n_grid: int) -> dict[str, object]:
    x_pos = torch.linspace(0.02, 0.98, n_grid, device=device, dtype=torch.float32).reshape(-1, 1)
    x_all = torch.linspace(-1.0, 1.0, 2 * n_grid, device=device, dtype=torch.float32).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        y_pos = model(x_pos)
        y_neg = model(-x_pos)
        output_even = float(relative_defect(y_pos, y_neg, "even").mean().item())
        output_odd = float(relative_defect(y_pos, y_neg, "odd").mean().item())
        p_pos = collect_partials(model, model_kind, x_pos)
        p_neg = collect_partials(model, model_kind, -x_pos)
        p_all = collect_partials(model, model_kind, x_all)
    layer_rows: list[dict[str, float | int]] = []
    per_channel_even: list[np.ndarray] = []
    per_channel_odd: list[np.ndarray] = []
    per_channel_energy: list[np.ndarray] = []
    per_channel_minima: list[np.ndarray] = []
    for layer_idx, (a, b, full_values) in enumerate(zip(p_pos, p_neg, p_all), start=1):
        even_values = relative_defect(a, b, "even").detach().cpu().numpy()
        odd_values = relative_defect(a, b, "odd").detach().cpu().numpy()
        energy_values = torch.mean(a ** 2, dim=0).detach().cpu().numpy()
        minima_values = strict_positive_minima(full_values.detach().cpu().numpy())
        per_channel_even.append(even_values)
        per_channel_odd.append(odd_values)
        per_channel_energy.append(energy_values)
        per_channel_minima.append(minima_values)
        layer_rows.append({
            "layer": layer_idx,
            "channels": int(even_values.shape[0]),
            "even_mean": float(np.mean(even_values)),
            "even_median": float(np.median(even_values)),
            "even_p90": float(np.quantile(even_values, 0.90)),
            "odd_mean": float(np.mean(odd_values)),
            "energy_mean": float(np.mean(energy_values)),
            "minima_mean": float(np.mean(minima_values)),
            "minima_p90": float(np.quantile(minima_values, 0.90)),
        })
    return {
        "output_even_defect": output_even,
        "output_odd_defect": output_odd,
        "layers": layer_rows,
        "per_channel_even": per_channel_even,
        "per_channel_odd": per_channel_odd,
        "per_channel_energy": per_channel_energy,
        "per_channel_minima": per_channel_minima,
    }


def mirror_distribution(model: nn.Module, model_kind: str) -> dict[str, np.ndarray | float]:
    if model_kind == "mmnn":
        first = model.fcs[0]
        next_weight = model.fcs[1].weight.detach().cpu().numpy()
    else:
        assert isinstance(model, FullRankMLP)
        first = model.hidden[0]
        next_weight = model.hidden[1].weight.detach().cpu().numpy() if len(model.hidden) > 1 else model.out.weight.detach().cpu().numpy()
    slopes = first.weight.detach().cpu().numpy().reshape(-1)
    biases = first.bias.detach().cpu().numpy().reshape(-1)
    mirror_idx = np.zeros_like(slopes, dtype=np.int64)
    mirror_dist = np.zeros_like(slopes, dtype=np.float64)
    mismatch_same = np.zeros_like(slopes, dtype=np.float64)
    mismatch_opposite = np.zeros_like(slopes, dtype=np.float64)
    signed_corr = np.zeros_like(slopes, dtype=np.float64)
    for j in range(slopes.shape[0]):
        distances = (slopes + slopes[j]) ** 2 + (biases - biases[j]) ** 2
        distances[j] = np.inf
        k = int(np.argmin(distances))
        cj = next_weight[:, j]
        ck = next_weight[:, k]
        denom = float(np.mean(cj ** 2 + ck ** 2) + EPS)
        mirror_idx[j] = k
        mirror_dist[j] = float(np.sqrt(distances[k]))
        mismatch_same[j] = float(np.mean((cj - ck) ** 2) / denom)
        mismatch_opposite[j] = float(np.mean((cj + ck) ** 2) / denom)
        signed_corr[j] = float(np.dot(cj, ck) / (np.linalg.norm(cj) * np.linalg.norm(ck) + EPS))
    close = mirror_dist <= np.quantile(mirror_dist, 0.20)
    return {
        "mirror_index": mirror_idx,
        "mirror_distance": mirror_dist,
        "mismatch_same": mismatch_same,
        "mismatch_opposite": mismatch_opposite,
        "signed_corr": signed_corr,
        "mirror_distance_mean": float(np.mean(mirror_dist)),
        "mirror_distance_p20": float(np.quantile(mirror_dist, 0.20)),
        "mismatch_same_mean": float(np.mean(mismatch_same)),
        "mismatch_same_best20": float(np.mean(mismatch_same[close])),
        "mismatch_opposite_mean": float(np.mean(mismatch_opposite)),
        "signed_corr_best20": float(np.mean(signed_corr[close])),
    }


def save_run_plots(out_dir: Path, cfg: Config, losses: list[float], function_metrics: dict[str, object], mirror: dict[str, np.ndarray | float]) -> None:
    plt.figure(figsize=(7, 4))
    plt.semilogy(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title(run_name(cfg))
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=220)
    plt.close()
    layer_rows = function_metrics["layers"]
    layers = [int(row["layer"]) for row in layer_rows]
    even = [float(row["even_mean"]) for row in layer_rows]
    minima = [float(row["minima_mean"]) for row in layer_rows]
    plt.figure(figsize=(7, 4))
    plt.plot(layers, even, marker="o", label="partial even defect")
    plt.yscale("log")
    plt.xlabel("partial layer")
    plt.ylabel("mean per-channel defect")
    plt.title("Input-space partial symmetry by layer")
    plt.tight_layout()
    plt.savefig(out_dir / "layerwise_partial_even_defect.png", dpi=220)
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.plot(layers, minima, marker="o")
    plt.xlabel("partial layer")
    plt.ylabel("mean strict positive minima")
    plt.title("Oscillatory complexity by layer")
    plt.tight_layout()
    plt.savefig(out_dir / "layerwise_minima.png", dpi=220)
    plt.close()
    all_even = np.concatenate(function_metrics["per_channel_even"]) if function_metrics["per_channel_even"] else np.array([])
    if all_even.size:
        plt.figure(figsize=(7, 4))
        plt.hist(np.log10(all_even + EPS), bins=40, alpha=0.85)
        plt.xlabel("log10 per-channel even defect")
        plt.ylabel("count")
        plt.title("Distribution of partial-function symmetry defects")
        plt.tight_layout()
        plt.savefig(out_dir / "partial_even_defect_distribution.png", dpi=220)
        plt.close()
    plt.figure(figsize=(7, 4))
    plt.hist(np.log10(np.asarray(mirror["mirror_distance"]) + EPS), bins=40, alpha=0.75, label="mirror distance")
    plt.hist(np.log10(np.asarray(mirror["mismatch_same"]) + EPS), bins=40, alpha=0.55, label="same-coeff mismatch")
    plt.xlabel("log10 value")
    plt.ylabel("count")
    plt.title("Weight-space mirror distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "weightspace_distribution.png", dpi=220)
    plt.close()


def save_fit_plot(out_dir: Path, cfg: Config, model: nn.Module, device: torch.device) -> None:
    x_np = np.linspace(-1.0, 1.0, 1000)
    y_np = target_sumcos(x_np, cfg.factor)
    x_t = torch.tensor(x_np.reshape(-1, 1), device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(x_t).detach().cpu().numpy().reshape(-1)
    plt.figure(figsize=(8, 4))
    plt.plot(x_np, y_np, label="target", linewidth=2)
    plt.plot(x_np, pred, label="prediction", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(run_name(cfg))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fit_target_prediction.png", dpi=220)
    plt.close()


def train_one(cfg: Config, overwrite: bool) -> dict[str, object]:
    out_dir = OUT_ROOT / run_name(cfg)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        with open(metrics_path) as f:
            return json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    x_np = np.linspace(-1.0, 1.0, cfg.n_train)
    y_np = target_sumcos(x_np, cfg.factor)
    x = torch.tensor(x_np.reshape(-1, 1), device=device, dtype=torch.float32)
    y = torch.tensor(y_np.reshape(-1, 1), device=device, dtype=torch.float32)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    current_lr = cfg.lr
    last_reduce = -1
    reductions: list[int] = []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(cfg.n_train, device=device)
        total = 0.0
        n_batches = 0
        for start in range(0, cfg.n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            n_batches += 1
        losses.append(total / max(1, n_batches))
        if epoch >= 50 and epoch - last_reduce >= 40 and len(losses) >= 20:
            recent = float(np.mean(losses[-10:]))
            previous = float(np.mean(losses[-20:-10]))
            if recent >= previous:
                current_lr *= 0.5
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                last_reduce = epoch
                reductions.append(epoch)
        if (epoch + 1) % max(1, cfg.epochs // 4) == 0:
            print(f"  {run_name(cfg)} epoch={epoch + 1}/{cfg.epochs} loss={losses[-1]:.4e} lr={current_lr:.2e}", flush=True)
    model.eval()
    with torch.no_grad():
        final_train = float(loss_fn(model(x), y).item())
    function_metrics = function_space_metrics(model, cfg.model_kind, device, n_grid=768)
    mirror = mirror_distribution(model, cfg.model_kind)
    np.savez_compressed(
        out_dir / "distributions.npz",
        **{f"partial_even_layer{i + 1}": arr for i, arr in enumerate(function_metrics["per_channel_even"])},
        **{f"partial_odd_layer{i + 1}": arr for i, arr in enumerate(function_metrics["per_channel_odd"])},
        **{f"partial_energy_layer{i + 1}": arr for i, arr in enumerate(function_metrics["per_channel_energy"])},
        **{f"partial_minima_layer{i + 1}": arr for i, arr in enumerate(function_metrics["per_channel_minima"])},
        mirror_distance=np.asarray(mirror["mirror_distance"]),
        mismatch_same=np.asarray(mirror["mismatch_same"]),
        mismatch_opposite=np.asarray(mirror["mismatch_opposite"]),
        signed_corr=np.asarray(mirror["signed_corr"]),
        losses=np.asarray(losses),
        lr_reductions=np.asarray(reductions, dtype=np.int64),
    )
    metrics: dict[str, object] = {
        **asdict(cfg),
        "name": run_name(cfg),
        "final_train_mse": final_train,
        "epochs_done": len(losses),
        "lr_reductions": reductions,
        "output_even_defect": function_metrics["output_even_defect"],
        "output_odd_defect": function_metrics["output_odd_defect"],
        "last_layer_even_mean": float(function_metrics["layers"][-1]["even_mean"]) if function_metrics["layers"] else 0.0,
        "last_layer_even_median": float(function_metrics["layers"][-1]["even_median"]) if function_metrics["layers"] else 0.0,
        "mean_layer_even": float(np.mean([float(row["even_mean"]) for row in function_metrics["layers"]])) if function_metrics["layers"] else 0.0,
        "mean_layer_minima": float(np.mean([float(row["minima_mean"]) for row in function_metrics["layers"]])) if function_metrics["layers"] else 0.0,
        "layers": function_metrics["layers"],
        **{k: v for k, v in mirror.items() if isinstance(v, float)},
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    torch.save(model.state_dict(), out_dir / "model_parameters.pth")
    save_run_plots(out_dir, cfg, losses, function_metrics, mirror)
    save_fit_plot(out_dir, cfg, model, device)
    return metrics


def config_grid(args: argparse.Namespace) -> list[Config]:
    seeds = parse_ints(args.seeds)
    widths = parse_ints(args.widths)
    ranks = parse_ints(args.ranks)
    depths = parse_ints(args.depths)
    configs: list[Config] = []
    for seed in seeds:
        for width in widths:
            for depth in depths:
                if "mlp" in args.models:
                    configs.append(Config("mlp", seed, width, width, depth, args.n_train, args.batch_size, args.epochs, args.lr, args.factor, args.momentum))
                if "mmnn" in args.models:
                    for rank in ranks:
                        configs.append(Config("mmnn", seed, width, rank, depth, args.n_train, args.batch_size, args.epochs, args.lr, args.factor, args.momentum))
    return configs


def write_summary(rows: list[dict[str, object]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    fields = [
        "name", "model_kind", "seed", "width", "rank", "depth", "n_train", "batch_size", "epochs",
        "final_train_mse", "output_even_defect", "last_layer_even_mean", "mean_layer_even",
        "mean_layer_minima", "mirror_distance_p20", "mismatch_same_best20", "signed_corr_best20",
    ]
    with open(OUT_ROOT / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    kinds = sorted({str(row["model_kind"]) for row in rows})
    plt.figure(figsize=(8, 5))
    for kind in kinds:
        values = [float(row["last_layer_even_mean"]) for row in rows if row["model_kind"] == kind]
        plt.hist(np.log10(np.asarray(values) + EPS), bins=24, alpha=0.55, label=kind)
    plt.xlabel("log10 last-layer partial even defect")
    plt.ylabel("config count")
    plt.title("Distribution across trained configs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "summary_last_layer_even_distribution.png", dpi=240)
    plt.close()
    plt.figure(figsize=(8, 5))
    for kind in kinds:
        x = [float(row["output_even_defect"]) for row in rows if row["model_kind"] == kind]
        y = [float(row["last_layer_even_mean"]) for row in rows if row["model_kind"] == kind]
        plt.scatter(x, y, label=kind, s=70, alpha=0.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("output even defect")
    plt.ylabel("last partial-layer even defect")
    plt.title("Output symmetry vs internal symmetry")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "summary_output_vs_partial_symmetry.png", dpi=240)
    plt.close()
    plt.figure(figsize=(8, 5))
    for kind in kinds:
        x = [float(row["mirror_distance_p20"]) for row in rows if row["model_kind"] == kind]
        y = [float(row["mismatch_same_best20"]) for row in rows if row["model_kind"] == kind]
        plt.scatter(x, y, label=kind, s=70, alpha=0.8)
    plt.yscale("log")
    plt.xlabel("close mirror-pair distance, p20")
    plt.ylabel("outgoing-weight mismatch, best 20%")
    plt.title("Weight-space mirror encoding across configs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "summary_weightspace_mirror_scatter.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["mmnn", "mlp"])
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--widths", default="384")
    parser.add_argument("--ranks", default="5,10,20")
    parser.add_argument("--depths", default="2,3")
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    configs = config_grid(args)
    print(f"running {len(configs)} configs -> {OUT_ROOT}", flush=True)
    rows: list[dict[str, object]] = []
    for index, cfg in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {run_name(cfg)}", flush=True)
        rows.append(train_one(cfg, overwrite=args.overwrite))
        write_summary(rows)
    write_summary(rows)
    print(f"done -> {OUT_ROOT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

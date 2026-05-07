#!/usr/bin/env python3
"""Quick symmetry experiments for MMNN/RF-LR style models versus full-rank MLPs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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


OUT_DIR = Path(__file__).resolve().parent / "results" / "symmetry_weightspace"


@dataclass(frozen=True)
class TrainConfig:
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

    def hidden_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        values: list[torch.Tensor] = []
        for layer in self.hidden:
            x = torch.relu(layer(x))
            values.append(x)
        return values


def target_sumcos(x: np.ndarray, factor: int) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    for k in range(1, factor + 1):
        y += np.cos(2.0 * np.pi * k * x)
    return y


def build_model(cfg: TrainConfig, device: torch.device) -> nn.Module:
    if cfg.model_kind == "mmnn":
        ranks = [1] + [cfg.rank] * cfg.depth + [1]
        widths = [cfg.width] * (cfg.depth + 1)
        return MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)
    if cfg.model_kind == "mlp":
        return FullRankMLP(width=cfg.width, depth=cfg.depth, device=device)
    raise ValueError(f"unknown model_kind: {cfg.model_kind}")


def get_partials(model: nn.Module, x: torch.Tensor, model_kind: str) -> list[torch.Tensor]:
    if model_kind == "mlp":
        assert isinstance(model, FullRankMLP)
        return model.hidden_activations(x)
    values: list[torch.Tensor] = []
    z = x
    depth = getattr(model, "depth")
    for j in range(depth - 1):
        z = model.fcs[2 * j](z)
        z = torch.relu(z)
        z = model.fcs[2 * j + 1](z)
        values.append(z)
    return values


def symmetry_defect(values_pos: torch.Tensor, values_neg: torch.Tensor) -> float:
    diff = values_pos - values_neg
    energy = torch.mean(values_pos ** 2) + 1e-12
    return float(torch.mean(diff ** 2).item() / float(energy.item()))


def layer_symmetry_defects(model: nn.Module, model_kind: str, device: torch.device, n_grid: int) -> list[float]:
    x = torch.linspace(0.02, 0.98, n_grid, device=device, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        pos = get_partials(model, x, model_kind)
        neg = get_partials(model, -x, model_kind)
    return [symmetry_defect(a, b) for a, b in zip(pos, neg)]


def output_symmetry_defect(model: nn.Module, device: torch.device, n_grid: int) -> float:
    x = torch.linspace(0.02, 0.98, n_grid, device=device, dtype=torch.float32).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        return symmetry_defect(model(x), model(-x))


def mirror_pair_stats(model: nn.Module, model_kind: str) -> dict[str, float]:
    if model_kind == "mmnn":
        first = model.fcs[0]
        next_weight = model.fcs[1].weight.detach().cpu().numpy()
    else:
        assert isinstance(model, FullRankMLP)
        first = model.hidden[0]
        next_weight = model.hidden[1].weight.detach().cpu().numpy() if len(model.hidden) > 1 else model.out.weight.detach().cpu().numpy()
    a = first.weight.detach().cpu().numpy().reshape(-1)
    b = first.bias.detach().cpu().numpy().reshape(-1)
    width = a.shape[0]
    mirror_distances: list[float] = []
    weight_mismatches: list[float] = []
    for j in range(width):
        distances = (a + a[j]) ** 2 + (b - b[j]) ** 2
        distances[j] = np.inf
        k = int(np.argmin(distances))
        coeff_j = next_weight[:, j]
        coeff_k = next_weight[:, k]
        denom = float(np.mean(coeff_j ** 2 + coeff_k ** 2) + 1e-12)
        mirror_distances.append(float(np.sqrt(distances[k])))
        weight_mismatches.append(float(np.mean((coeff_j - coeff_k) ** 2) / denom))
    cutoff = float(np.quantile(mirror_distances, 0.20))
    chosen = [i for i, d in enumerate(mirror_distances) if d <= cutoff]
    return {
        "mirror_distance_mean": float(np.mean(mirror_distances)),
        "mirror_distance_p20": cutoff,
        "mirror_weight_mismatch_mean": float(np.mean(weight_mismatches)),
        "mirror_weight_mismatch_best20": float(np.mean([weight_mismatches[i] for i in chosen])),
    }


def train_one(cfg: TrainConfig, out_dir: Path) -> dict[str, float | int | str | list[float]]:
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
    optimizer = optim.SGD((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, momentum=0.0)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    current_lr = cfg.lr
    last_reduce = -1
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(cfg.n_train, device=device)
        total = 0.0
        batches = 0
        for start in range(0, cfg.n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        losses.append(total / max(1, batches))
        if epoch >= 40 and epoch - last_reduce >= 30 and len(losses) >= 20:
            recent = float(np.mean(losses[-10:]))
            previous = float(np.mean(losses[-20:-10]))
            if recent >= previous:
                current_lr *= 0.5
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                last_reduce = epoch
    model.eval()
    with torch.no_grad():
        final_train = float(loss_fn(model(x), y).item())
    layer_defects = layer_symmetry_defects(model, cfg.model_kind, device, n_grid=512)
    result: dict[str, float | int | str | list[float]] = {
        "model_kind": cfg.model_kind,
        "seed": cfg.seed,
        "width": cfg.width,
        "rank": cfg.rank,
        "depth": cfg.depth,
        "n_train": cfg.n_train,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "final_train_mse": final_train,
        "output_symmetry_defect": output_symmetry_defect(model, device, n_grid=512),
        "mean_partial_symmetry_defect": float(np.mean(layer_defects)) if layer_defects else 0.0,
        "layer_partial_symmetry_defects": layer_defects,
        **mirror_pair_stats(model, cfg.model_kind),
    }
    torch.save(model.state_dict(), out_dir / f"{cfg.model_kind}_seed{cfg.seed}.pth")
    with open(out_dir / f"{cfg.model_kind}_seed{cfg.seed}_losses.json", "w") as f:
        json.dump({"config": cfg.__dict__, "losses": losses, "metrics": result}, f, indent=2)
    save_partial_heatmap(model, cfg.model_kind, device, out_dir / f"partial_symmetry_heatmap_{cfg.model_kind}_seed{cfg.seed}.png")
    return result


def save_partial_heatmap(model: nn.Module, model_kind: str, device: torch.device, path: Path) -> None:
    x = torch.linspace(0.02, 0.98, 400, device=device, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        pos = get_partials(model, x, model_kind)
        neg = get_partials(model, -x, model_kind)
    if not pos:
        return
    residual = (pos[-1] - neg[-1]).detach().cpu().numpy()
    energy = np.mean(pos[-1].detach().cpu().numpy() ** 2, axis=0)
    order = np.argsort(-energy)[: min(16, residual.shape[1])]
    img = residual[:, order].T
    plt.figure(figsize=(8, 4))
    vmax = float(np.percentile(np.abs(img), 98)) + 1e-12
    plt.imshow(img, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=[0.02, 0.98, len(order), 1])
    plt.colorbar(label="partial(x) - partial(-x)")
    plt.xlabel("x")
    plt.ylabel("top-energy channel")
    plt.title(f"Partial symmetry residual, {model_kind}")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_summary_plots(rows: list[dict[str, float | int | str | list[float]]], out_dir: Path) -> None:
    labels = [f"{r['model_kind']} s{r['seed']}" for r in rows]
    out_sym = [float(r["output_symmetry_defect"]) for r in rows]
    partial_sym = [float(r["mean_partial_symmetry_defect"]) for r in rows]
    mirror = [float(r["mirror_weight_mismatch_best20"]) for r in rows]
    x = np.arange(len(rows))
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - 0.25, out_sym, width=0.25, label="output")
    plt.bar(x, partial_sym, width=0.25, label="partials")
    plt.bar(x + 0.25, mirror, width=0.25, label="mirror weights")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("relative defect, log scale")
    plt.title("Symmetry defects in function space and weight space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "symmetry_defects_bar.png", dpi=240)
    plt.close()
    plt.figure(figsize=(6, 4.5))
    for kind in sorted({str(r["model_kind"]) for r in rows}):
        xs = [float(r["mirror_distance_p20"]) for r in rows if r["model_kind"] == kind]
        ys = [float(r["mirror_weight_mismatch_best20"]) for r in rows if r["model_kind"] == kind]
        plt.scatter(xs, ys, label=kind, s=70)
    plt.yscale("log")
    plt.xlabel("20% mirror-pair distance in first layer")
    plt.ylabel("best-20% outgoing weight mismatch")
    plt.title("Weight-space mirror-pair encoding")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mirror_pair_weightspace.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--factor", type=int, default=3)
    args = parser.parse_args()
    epochs = 300 if args.quick else args.epochs
    seeds = args.seeds[:1] if args.quick else args.seeds
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | list[float]]] = []
    for seed in seeds:
        for model_kind in ["mmnn", "mlp"]:
            cfg = TrainConfig(
                model_kind=model_kind,
                seed=seed,
                width=args.width,
                rank=args.rank,
                depth=args.depth,
                n_train=args.n_train,
                batch_size=args.batch_size,
                epochs=epochs,
                lr=args.lr,
                factor=args.factor,
            )
            print(f"train {model_kind} seed={seed} epochs={epochs}")
            rows.append(train_one(cfg, OUT_DIR))
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
        fields = [k for k in rows[0].keys() if k != "layer_partial_symmetry_defects"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    save_summary_plots(rows, OUT_DIR)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

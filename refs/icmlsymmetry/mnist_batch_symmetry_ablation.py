#!/usr/bin/env python3
"""MNIST batch-size ablation with representation and weight-space symmetry metrics."""
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.table.mmnn_vs import MMNN  # noqa: E402


OUT_ROOT = Path(__file__).resolve().parent / "results" / "mnist_batch_symmetry"
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    model_kind: str
    seed: int
    optimizer: str
    batch_size: int
    full_batch: bool
    epochs: int
    lr: float
    width: int
    rank: int
    depth: int
    train_subset: int
    test_subset: int
    momentum: float


class MLP(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 784
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def partials(self, x: torch.Tensor) -> list[torch.Tensor]:
        values: list[torch.Tensor] = []
        z = x
        for layer in self.layers:
            z = layer(z)
            if isinstance(layer, nn.ReLU):
                values.append(z)
        return values


def parse_batch_sizes(value: str, train_size: int) -> list[tuple[int, bool]]:
    out: list[tuple[int, bool]] = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"full", "fullbatch", "all"}:
            out.append((train_size, True))
        else:
            out.append((int(token), False))
    return out


def run_name(cfg: Config) -> str:
    bs = "full" if cfg.full_batch else str(cfg.batch_size)
    return f"{cfg.model_kind}_{cfg.optimizer}_seed{cfg.seed}_bs{bs}_W{cfg.width}_r{cfg.rank}_L{cfg.depth}_ep{cfg.epochs}_N{cfg.train_subset}"


def get_data(data_dir: Path, train_subset: int, test_subset: int, seed: int) -> tuple[Subset, Subset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.flatten()),
    ])
    train = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)
    train_indices = rng.permutation(len(train))[: min(train_subset, len(train))]
    test_indices = rng.permutation(len(test))[: min(test_subset, len(test))]
    return Subset(train, train_indices.tolist()), Subset(test, test_indices.tolist())


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    if cfg.model_kind == "mlp":
        return MLP(width=cfg.width, depth=cfg.depth).to(device)
    if cfg.model_kind == "mmnn":
        ranks = [784] + [cfg.rank] * max(0, cfg.depth - 1) + [10]
        widths = [cfg.width] * cfg.depth
        return MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True, factorize_first_rank=128).to(device)
    raise ValueError(f"unknown model_kind={cfg.model_kind}")


def collect_partials(model: nn.Module, model_kind: str, x: torch.Tensor) -> list[torch.Tensor]:
    if model_kind == "mlp":
        assert isinstance(model, MLP)
        return model.partials(x)
    values: list[torch.Tensor] = []
    z = x
    depth = int(getattr(model, "depth"))
    for j in range(depth - 1):
        z = model.fcs[2 * j](z)
        z = torch.relu(z)
        z = model.fcs[2 * j + 1](z)
        values.append(z)
    return values


def transform_flat(x: torch.Tensor, name: str) -> torch.Tensor:
    img = x.reshape(-1, 1, 28, 28)
    if name == "hflip":
        y = torch.flip(img, dims=[3])
    elif name == "vflip":
        y = torch.flip(img, dims=[2])
    elif name == "rot180":
        y = torch.flip(img, dims=[2, 3])
    elif name == "shift_right":
        y = torch.zeros_like(img)
        y[:, :, :, 1:] = img[:, :, :, :-1]
    elif name == "shift_down":
        y = torch.zeros_like(img)
        y[:, :, 1:, :] = img[:, :, :-1, :]
    else:
        raise ValueError(f"unknown transform={name}")
    return y.reshape(x.shape[0], -1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return {"acc": correct / max(1, total), "loss": loss_sum / max(1, total)}


def train_one(cfg: Config, data_dir: Path, overwrite: bool) -> dict[str, object]:
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
    train_set, test_set = get_data(data_dir, cfg.train_subset, cfg.test_subset, cfg.seed)
    effective_bs = len(train_set) if cfg.full_batch else cfg.batch_size
    train_loader = DataLoader(train_set, batch_size=effective_bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=min(512, len(test_set)), shuffle=False, num_workers=0)
    model = build_model(cfg, device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adam":
        optimizer = optim.Adam(trainable_params, lr=cfg.lr)
    elif cfg.optimizer == "sgd":
        optimizer = optim.SGD(trainable_params, lr=cfg.lr, momentum=cfg.momentum)
    else:
        raise ValueError(f"unknown optimizer={cfg.optimizer}")
    losses: list[float] = []
    accs: list[float] = []
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * int(y.numel())
            n_seen += int(y.numel())
        losses.append(total / max(1, n_seen))
        if (epoch + 1) % max(1, cfg.epochs // 5) == 0 or epoch == cfg.epochs - 1:
            test_metrics = evaluate(model, test_loader, device)
            accs.append(float(test_metrics["acc"]))
            print(f"  {run_name(cfg)} epoch={epoch + 1}/{cfg.epochs} loss={losses[-1]:.4f} test_acc={test_metrics['acc']:.4f}", flush=True)
    train_metrics = evaluate(model, train_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    symmetry = symmetry_metrics(model, cfg.model_kind, test_loader, device)
    weights = weight_symmetry_metrics(model, cfg.model_kind)
    metrics: dict[str, object] = {
        **asdict(cfg),
        "name": run_name(cfg),
        "effective_batch_size": effective_bs,
        "nparams": int(sum(p.numel() for p in model.parameters())),
        "ntrainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "final_train_loss": float(train_metrics["loss"]),
        "final_train_acc": float(train_metrics["acc"]),
        "final_test_loss": float(test_metrics["loss"]),
        "final_test_acc": float(test_metrics["acc"]),
        **symmetry,
        **weights,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump({"losses": losses, "test_acc_checkpoints": accs}, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model_parameters.pth")
    save_run_plots(out_dir, cfg, losses, metrics)
    return metrics


def symmetry_metrics(model: nn.Module, model_kind: str, loader: DataLoader, device: torch.device) -> dict[str, float]:
    transforms_to_test = ["hflip", "vflip", "rot180", "shift_right", "shift_down"]
    output_values: dict[str, list[float]] = {name: [] for name in transforms_to_test}
    partial_values: dict[str, list[float]] = {name: [] for name in transforms_to_test}
    consistency_values: dict[str, list[float]] = {name: [] for name in transforms_to_test}
    model.eval()
    with torch.no_grad():
        for batch_idx, (x_cpu, _) in enumerate(loader):
            if batch_idx >= 8:
                break
            x = x_cpu.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            base_partials = collect_partials(model, model_kind, x)
            for name in transforms_to_test:
                xt = transform_flat(x, name)
                logits_t = model(xt)
                pred_t = logits_t.argmax(dim=1)
                diff = torch.mean((logits - logits_t) ** 2) / (torch.mean(logits ** 2) + EPS)
                output_values[name].append(float(diff.item()))
                consistency_values[name].append(float((pred == pred_t).float().mean().item()))
                part_t = collect_partials(model, model_kind, xt)
                defects: list[float] = []
                for a, b in zip(base_partials, part_t):
                    energy = torch.mean(a ** 2, dim=0)
                    mask = energy >= torch.quantile(energy, 0.50)
                    if torch.any(mask):
                        d = torch.mean((a[:, mask] - b[:, mask]) ** 2) / (torch.mean(a[:, mask] ** 2) + EPS)
                        defects.append(float(d.item()))
                partial_values[name].append(float(np.mean(defects)) if defects else 0.0)
    out: dict[str, float] = {}
    for name in transforms_to_test:
        out[f"logit_defect_{name}"] = float(np.mean(output_values[name]))
        out[f"prediction_consistency_{name}"] = float(np.mean(consistency_values[name]))
        out[f"partial_defect_{name}"] = float(np.mean(partial_values[name]))
    out["logit_defect_mean"] = float(np.mean([out[f"logit_defect_{name}"] for name in transforms_to_test]))
    out["partial_defect_mean"] = float(np.mean([out[f"partial_defect_{name}"] for name in transforms_to_test]))
    out["prediction_consistency_mean"] = float(np.mean([out[f"prediction_consistency_{name}"] for name in transforms_to_test]))
    return out


def weight_symmetry_metrics(model: nn.Module, model_kind: str) -> dict[str, float]:
    weight_matrix = first_image_weight_matrix(model, model_kind)
    if weight_matrix.size == 0:
        return {}
    images = weight_matrix.reshape(weight_matrix.shape[0], 28, 28)
    transforms_np = {
        "hflip": np.flip(images, axis=2),
        "vflip": np.flip(images, axis=1),
        "rot180": np.flip(images, axis=(1, 2)),
    }
    out: dict[str, float] = {}
    for name, transformed in transforms_np.items():
        self_defect = np.mean((images - transformed) ** 2, axis=(1, 2)) / (np.mean(images ** 2, axis=(1, 2)) + EPS)
        pair_distances = nearest_filter_distances(images, transformed)
        out[f"first_weight_self_{name}_mean"] = float(np.mean(self_defect))
        out[f"first_weight_self_{name}_median"] = float(np.median(self_defect))
        out[f"first_weight_pair_{name}_mean"] = float(np.mean(pair_distances))
        out[f"first_weight_pair_{name}_p20"] = float(np.quantile(pair_distances, 0.20))
    out["first_weight_self_symmetry_mean"] = float(np.mean([out[f"first_weight_self_{name}_mean"] for name in transforms_np]))
    out["first_weight_pair_symmetry_mean"] = float(np.mean([out[f"first_weight_pair_{name}_mean"] for name in transforms_np]))
    return out


def first_image_weight_matrix(model: nn.Module, model_kind: str) -> np.ndarray:
    if model_kind == "mlp":
        assert isinstance(model, MLP)
        first = model.layers[0]
        assert isinstance(first, nn.Linear)
        return first.weight.detach().cpu().numpy()
    first = model.fcs[0]
    if hasattr(first, "lin1"):
        lin1 = first.lin1.weight.detach().cpu().numpy()
        lin2 = first.lin2.weight.detach().cpu().numpy()
        return lin2 @ lin1
    return first.weight.detach().cpu().numpy()


def nearest_filter_distances(images: np.ndarray, transformed: np.ndarray) -> np.ndarray:
    flat = images.reshape(images.shape[0], -1)
    tflat = transformed.reshape(transformed.shape[0], -1)
    norms = np.mean(flat ** 2, axis=1) + EPS
    distances = np.zeros(images.shape[0], dtype=np.float64)
    for idx in range(images.shape[0]):
        diff = flat - tflat[idx][None, :]
        d = np.mean(diff ** 2, axis=1) / norms
        distances[idx] = float(np.min(d))
    return distances


def save_run_plots(out_dir: Path, cfg: Config, losses: list[float], metrics: dict[str, object]) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("train CE")
    plt.title(run_name(cfg))
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=220)
    plt.close()
    names = ["hflip", "vflip", "rot180", "shift_right", "shift_down"]
    logit = [float(metrics[f"logit_defect_{name}"]) for name in names]
    partial = [float(metrics[f"partial_defect_{name}"]) for name in names]
    x = np.arange(len(names))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.2, logit, width=0.4, label="logits")
    plt.bar(x + 0.2, partial, width=0.4, label="partials")
    plt.yscale("log")
    plt.xticks(x, names, rotation=20)
    plt.ylabel("relative transform defect")
    plt.title("MNIST input-transform defects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "input_transform_defects.png", dpi=220)
    plt.close()


def write_summary(rows: list[dict[str, object]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    fields = [
        "name", "model_kind", "optimizer", "seed", "effective_batch_size", "full_batch", "width", "rank", "depth",
        "epochs", "train_subset", "final_test_acc", "final_test_loss", "ntrainable",
        "logit_defect_mean", "partial_defect_mean", "prediction_consistency_mean",
        "first_weight_self_symmetry_mean", "first_weight_pair_symmetry_mean",
    ]
    with open(OUT_ROOT / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    save_summary_plots(rows)


def save_summary_plots(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    plt.figure(figsize=(8, 5))
    for kind, marker in [("mlp", "s"), ("mmnn", "o")]:
        for opt in sorted({str(r.get("optimizer", "sgd")) for r in rows}):
            xs = [float(r["effective_batch_size"]) for r in rows if r["model_kind"] == kind and r.get("optimizer", "sgd") == opt]
            ys = [float(r["final_test_acc"]) for r in rows if r["model_kind"] == kind and r.get("optimizer", "sgd") == opt]
            if xs:
                plt.scatter(xs, ys, label=f"{kind}-{opt}", marker=marker, s=70)
    plt.xscale("log")
    plt.xlabel("batch size")
    plt.ylabel("test accuracy")
    plt.title("MNIST accuracy vs batch size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "mnist_accuracy_vs_batch.png", dpi=240)
    plt.close()
    plt.figure(figsize=(8, 5))
    for kind, marker in [("mlp", "s"), ("mmnn", "o")]:
        for opt in sorted({str(r.get("optimizer", "sgd")) for r in rows}):
            xs = [float(r["effective_batch_size"]) for r in rows if r["model_kind"] == kind and r.get("optimizer", "sgd") == opt]
            ys = [float(r["partial_defect_mean"]) for r in rows if r["model_kind"] == kind and r.get("optimizer", "sgd") == opt]
            if xs:
                plt.scatter(xs, ys, label=f"{kind}-{opt}", marker=marker, s=70)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("batch size")
    plt.ylabel("mean partial transform defect")
    plt.title("MNIST internal symmetry vs batch size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "mnist_partial_defect_vs_batch.png", dpi=240)
    plt.close()
    plt.figure(figsize=(8, 5))
    for kind, marker in [("mlp", "s"), ("mmnn", "o")]:
        for opt in sorted({str(r.get("optimizer", "sgd")) for r in rows}):
            xs = [float(r["logit_defect_mean"]) for r in rows if r["model_kind"] == kind and r.get("optimizer", "sgd") == opt]
            ys = [float(r["partial_defect_mean"]) for r in rows if r["model_kind"] == kind and r.get("optimizer", "sgd") == opt]
            if xs:
                plt.scatter(xs, ys, label=f"{kind}-{opt}", marker=marker, s=70)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("mean logit transform defect")
    plt.ylabel("mean partial transform defect")
    plt.title("MNIST output-space vs representation-space symmetry")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "mnist_output_vs_partial_defect.png", dpi=240)
    plt.close()
    plt.figure(figsize=(8, 5))
    for kind, marker in [("mlp", "s"), ("mmnn", "o")]:
        for opt in sorted({str(r.get("optimizer", "sgd")) for r in rows}):
            xs = [float(r["first_weight_pair_symmetry_mean"]) for r in rows if r["model_kind"] == kind and r.get("optimizer", "sgd") == opt]
            ys = [float(r["partial_defect_mean"]) for r in rows if r["model_kind"] == kind and r.get("optimizer", "sgd") == opt]
            if xs:
                plt.scatter(xs, ys, label=f"{kind}-{opt}", marker=marker, s=70)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("first-layer nearest transformed-filter defect")
    plt.ylabel("mean partial transform defect")
    plt.title("MNIST weight-space vs representation symmetry")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "mnist_weightspace_vs_partial_defect.png", dpi=240)
    plt.close()


def config_grid(args: argparse.Namespace) -> list[Config]:
    train_size = args.train_subset
    batch_specs = parse_batch_sizes(args.batch_sizes, train_size)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    models = set(args.models)
    configs: list[Config] = []
    for seed in seeds:
        for batch_size, full_batch in batch_specs:
            if "mlp" in models:
                configs.append(Config("mlp", seed, args.optimizer, batch_size, full_batch, args.epochs, args.lr, args.width, args.width, args.depth, args.train_subset, args.test_subset, args.momentum))
            if "mmnn" in models:
                for rank in ranks:
                    configs.append(Config("mmnn", seed, args.optimizer, batch_size, full_batch, args.epochs, args.lr, args.width, rank, args.depth, args.train_subset, args.test_subset, args.momentum))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models", nargs="*", default=["mlp", "mmnn"])
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--batch-sizes", default="1,8,64,512,full")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--rank", dest="ranks", default="10,25")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    configs = config_grid(args)
    print(f"running {len(configs)} MNIST configs -> {OUT_ROOT}", flush=True)
    rows: list[dict[str, object]] = []
    for index, cfg in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {run_name(cfg)}", flush=True)
        rows.append(train_one(cfg, args.data_dir, args.overwrite))
        write_summary(rows)
    write_summary(rows)
    print(f"done -> {OUT_ROOT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ICML rebuttal: multi-seed runs for frozen RF (fixWb) MMNN — variance over random feature init.

Examples:
  # MNIST, 5 seeds, ranks 15 and 25 (matches paper-style table)
  python experiments/table/icml_rebuttal_experiments.py --mnist --seeds 0 1 2 3 4 --ranks 15 25

  # Quick smoke test
  python experiments/table/icml_rebuttal_experiments.py --mnist --quick --seeds 0 1 2

  # CIFAR-10 (flattened 32x32x3)
  python experiments/table/icml_rebuttal_experiments.py --cifar10 --seeds 0 1 2 3 4 --ranks 15 25
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_mmnn_vs_mlp import run_mmnn


def get_mnist_loaders(data_dir: Path, batch_size: int, num_workers: int = 0):
    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.flatten()),
    ])
    train = datasets.MNIST(str(data_dir), train=True, download=True, transform=tr)
    test = datasets.MNIST(str(data_dir), train=False, download=True, transform=tr)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def get_cifar10_loaders(data_dir: Path, batch_size: int, num_workers: int = 0):
    """Flatten CIFAR-10 to 3072; standard normalization."""
    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.Lambda(lambda x: x.flatten()),
    ])
    train = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=tr)
    test = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=tr)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
    return m, math.sqrt(v) if len(xs) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="ICML rebuttal: multi-seed MMNN (frozen RF)")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/table/icml_rebuttal_runs"))
    ap.add_argument("--mnist", action="store_true", help="Run MNIST")
    ap.add_argument("--cifar10", action="store_true", help="Run CIFAR-10")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Seeds (RF / full init)")
    ap.add_argument("--ranks", nargs="+", type=int, default=[15, 25], help="Hidden rank R per MMNN")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--num-hidden", type=int, default=2, help="Number of MMNN blocks")
    ap.add_argument("--factorize-first", type=int, default=128, help="0 to disable")
    ap.add_argument("--quick", action="store_true", help="Fewer epochs for smoke test")
    args = ap.parse_args()

    if not args.mnist and not args.cifar10:
        args.mnist = True

    if args.quick:
        args.epochs = min(args.epochs, 5)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factorize_first = args.factorize_first if args.factorize_first > 0 else None

    all_runs: list[dict] = []

    def run_dataset(name: str, in_dim: int, train_loader: DataLoader, test_loader: DataLoader) -> None:
        for R in args.ranks:
            ranks = [in_dim] + [R] * max(0, args.num_hidden - 1) + [10]
            widths = [args.hidden] * args.num_hidden
            row_accs: list[float] = []
            row_losses: list[float] = []
            for seed in tqdm(args.seeds, desc=f"{name} R={R}"):
                desc = f"{name} MMNN R={R} seed={seed}"
                r, _model = run_mmnn(
                    device,
                    train_loader,
                    test_loader,
                    num_epochs=args.epochs,
                    lr=args.lr,
                    ranks=ranks,
                    widths=widths,
                    fix_wb=True,
                    resnet=False,
                    seed=seed,
                    desc=desc,
                    factorize_first_rank=factorize_first,
                )
                row_accs.append(r["final_test_acc"])
                row_losses.append(r["final_test_loss"])
                all_runs.append({
                    "dataset": name,
                    "rank": R,
                    "seed": seed,
                    "final_test_acc": r["final_test_acc"],
                    "final_test_loss": r["final_test_loss"],
                    "nparams": r["nparams"],
                    "ntrainable": r["ntrainable"],
                    "epochs": args.epochs,
                })
            m_acc, s_acc = _mean_std(row_accs)
            m_loss, s_loss = _mean_std(row_losses)
            print(
                f"  [{name}] R={R}  test_acc: {m_acc:.4f} ± {s_acc:.4f}  "
                f"test_loss: {m_loss:.4f} ± {s_loss:.4f}  (n={len(args.seeds)})"
            )

    if args.mnist:
        train_loader, test_loader = get_mnist_loaders(args.data_dir, args.batch_size)
        run_dataset("MNIST", 784, train_loader, test_loader)

    if args.cifar10:
        train_loader, test_loader = get_cifar10_loaders(args.data_dir, args.batch_size)
        run_dataset("CIFAR10", 3072, train_loader, test_loader)

    summary_path = args.out_dir / "icml_rebuttal_seed_sweep.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "args": {
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "hidden": args.hidden,
                    "num_hidden": args.num_hidden,
                    "factorize_first": factorize_first,
                    "seeds": args.seeds,
                    "ranks": args.ranks,
                },
                "runs": all_runs,
            },
            f,
            indent=2,
        )
    print(f"Saved {summary_path}")

    # aggregate table by dataset x rank
    by_key: dict[tuple[str, int], list[float]] = {}
    for run in all_runs:
        k = (run["dataset"], run["rank"])
        by_key.setdefault(k, []).append(run["final_test_acc"])

    lines = [
        "# ICML rebuttal — frozen RF MMNN, multi-seed",
        "",
        "| Dataset | Rank R | mean test acc | std | seeds |",
        "|---------|--------|---------------|-----|-------|",
    ]
    for (ds, R), accs in sorted(by_key.items()):
        m, s = _mean_std(accs)
        lines.append(
            f"| {ds} | {R} | {m:.4f} | {s:.4f} | {len(accs)} |"
        )
    md_path = args.out_dir / "icml_rebuttal_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()

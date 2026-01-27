#!/usr/bin/env python3
"""
MNIST benchmark: MLP baseline vs low-rank MMNN with frozen features.

- MLP: standard ReLU MLP, all parameters trained.
- MMNN counterpart: very low rank (R), fixWb=True (frozen rank→width projections;
  only width→rank matrices are trained).

Trains both on MNIST with CrossEntropyLoss, saves comparison plot and models.

Outputs in --out-dir:
  - mlp.pt, mmnn_r{R}.pt (for each --mmnn-ranks): {"state_dict": ..., "config": {...}}
  - results.json, histories.json, mnist_mmnn_vs_mlp.png, results.md
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# we add script dir so we can import mmnn_vs from the same package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmnn_vs import MMNN


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard ReLU MLP: Linear -> ReLU (x num_hidden) -> Linear."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_hidden: int):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# MNIST data
# ---------------------------------------------------------------------------

def get_mnist_loaders(data_dir: Path, batch_size: int, num_workers: int = 0):
    """MNIST train and test loaders; flatten 28x28 to 784 and normalize."""
    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.flatten()),
    ])
    train = datasets.MNIST(str(data_dir), train=True, download=True, transform=tr)
    test = datasets.MNIST(str(data_dir), train=False, download=True, transform=tr)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Accuracy and mean CrossEntropy loss on the given loader."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total), loss_sum / max(1, total)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One epoch; returns mean training loss."""
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
    return loss_sum / max(1, n)


def run_mmnn(
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    lr: float,
    ranks: list[int],
    widths: list[int],
    fix_wb: bool,
    resnet: bool,
    seed: int,
    desc: str = "MMNN",
) -> tuple[dict, nn.Module]:
    """Train MMNN and return (history and final metrics, trained model)."""
    torch.manual_seed(seed)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=resnet, fixWb=fix_wb)
    model = model.to(device)
    nparams = sum(p.numel() for p in model.parameters())
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"loss": [], "test_acc": [], "test_loss": []}

    for ep in tqdm(range(num_epochs), desc=desc, unit="ep"):
        loss = train_epoch(model, train_loader, optimizer, device)
        acc, tloss = evaluate(model, test_loader, device)
        hist["loss"].append(float(loss))
        hist["test_acc"].append(float(acc))
        hist["test_loss"].append(float(tloss))

    acc_f, tloss_f = evaluate(model, test_loader, device)
    res = {
        "history": hist,
        "final_test_acc": float(acc_f),
        "final_test_loss": float(tloss_f),
        "nparams": nparams,
        "ntrainable": ntrain,
        "arch": "MMNN",
        "ranks": ranks,
        "widths": widths,
        "fix_wb": fix_wb,
    }
    return res, model


def run_mlp(
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    in_dim: int,
    out_dim: int,
    hidden: int,
    num_hidden: int,
    num_epochs: int,
    lr: float,
    seed: int,
) -> tuple[dict, nn.Module]:
    """Train MLP and return (history and final metrics, trained model)."""
    torch.manual_seed(seed)
    model = MLP(in_dim, hidden, out_dim, num_hidden).to(device)
    nparams = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"loss": [], "test_acc": [], "test_loss": []}

    for ep in tqdm(range(num_epochs), desc="MLP", unit="ep"):
        loss = train_epoch(model, train_loader, optimizer, device)
        acc, tloss = evaluate(model, test_loader, device)
        hist["loss"].append(float(loss))
        hist["test_acc"].append(float(acc))
        hist["test_loss"].append(float(tloss))

    acc_f, tloss_f = evaluate(model, test_loader, device)
    res = {
        "history": hist,
        "final_test_acc": float(acc_f),
        "final_test_loss": float(tloss_f),
        "nparams": nparams,
        "ntrainable": nparams,
        "arch": "MLP",
        "hidden": hidden,
        "num_hidden": num_hidden,
    }
    return res, model


# ---------------------------------------------------------------------------
# Plot and main
# ---------------------------------------------------------------------------

def _label(r: dict) -> str:
    """Build legend label: MLP (P=..) or MMNN R=.. fixWb? (P=.., train=..)."""
    if r["arch"] == "MLP":
        return f"MLP (P={r['nparams']:,})"
    R = r.get("ranks", [0, 0])[1] if len(r.get("ranks", [])) > 1 else "?"
    if r.get("fix_wb") and r.get("ntrainable", r["nparams"]) < r["nparams"]:
        return f"MMNN R={R} fixWb (P={r['nparams']:,}, train={r['ntrainable']:,})"
    return f"MMNN R={R} (P={r['nparams']:,})"


def plot_comparison(results: list[dict], out_path: Path) -> None:
    """Plot loss and test accuracy for each run."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, r in enumerate(results):
        h = r["history"]
        lab = _label(r)
        c = colors[i % len(colors)]
        epochs = range(1, len(h["loss"]) + 1)
        ax1.plot(epochs, h["loss"], color=c, label=lab, alpha=0.9)
        ax2.plot(epochs, [a * 100 for a in h["test_acc"]], color=c, label=lab, alpha=0.9)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax1.set_title("MNIST: MLP baseline vs low-rank MMNN (frozen features)")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test accuracy (%)")
    ax2.set_title("MNIST: MLP baseline vs low-rank MMNN (frozen features)")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description="MNIST MMNN vs MLP")
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST data directory")
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/table/mnist_mmnn_vs_mlp"), help="Output directory")
    ap.add_argument("--epochs", type=int, default=30, help="Epochs per model")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--mmnn-ranks", nargs="+", type=int, default=[5, 10], help="MMNN hidden rank(s), e.g. 5 10 (each with fixWb/random features)")
    ap.add_argument("--hidden", type=int, default=512, help="Hidden size (MLP and MMNN width)")
    ap.add_argument("--num-hidden", type=int, default=2, help="Number of hidden layers (MLP) / blocks (MMNN)")
    ap.add_argument("--no-fix-wb", action="store_true", help="MMNN: do not freeze features (default: fixWb=True)")
    ap.add_argument("--skip-mlp", action="store_true", help="Only run MMNN")
    ap.add_argument("--skip-mmnn", action="store_true", help="Only run MLP")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, test_loader = get_mnist_loaders(args.data_dir, args.batch_size)
    in_dim, out_dim = 784, 10

    results = []
    fix_wb = not args.no_fix_wb

    # 1) MLP baseline: full MLP, all parameters trained
    if not args.skip_mlp:
        r, model = run_mlp(
            device, train_loader, test_loader,
            in_dim=in_dim, out_dim=out_dim, hidden=args.hidden, num_hidden=args.num_hidden,
            num_epochs=args.epochs, lr=args.lr, seed=args.seed,
        )
        torch.save({
            "state_dict": model.state_dict(),
            "config": {"in_dim": in_dim, "out_dim": out_dim, "hidden": args.hidden, "num_hidden": args.num_hidden},
        }, args.out_dir / "mlp.pt")
        print(f"  saved {args.out_dir / 'mlp.pt'}")
        results.append(r)
        print(f"  MLP:  test acc={r['final_test_acc']:.4f}, test loss={r['final_test_loss']:.4f}, params={r['nparams']:,}")

    # 2) MMNN counterpart(s): very low rank(s), random/frozen features (fixWb=True by default)
    if not args.skip_mmnn:
        for R in args.mmnn_ranks:
            ranks = [in_dim] + [R] * max(0, args.num_hidden - 1) + [out_dim]
            widths = [args.hidden] * args.num_hidden
            desc = f"MMNN R={R} fixWb" if fix_wb else f"MMNN R={R}"
            r, model = run_mmnn(
                device, train_loader, test_loader,
                num_epochs=args.epochs, lr=args.lr,
                ranks=ranks, widths=widths, fix_wb=fix_wb, resnet=False, seed=args.seed, desc=desc,
            )
            torch.save({
                "state_dict": model.state_dict(),
                "config": {"ranks": ranks, "widths": widths, "fixWb": fix_wb, "ResNet": False},
            }, args.out_dir / f"mmnn_r{R}.pt")
            print(f"  saved {args.out_dir / f'mmnn_r{R}.pt'}")
            results.append(r)
            print(f"  MMNN R={R} fixWb: test acc={r['final_test_acc']:.4f}, test loss={r['final_test_loss']:.4f}, params={r['nparams']:,} (trainable={r['ntrainable']:,})")

    # we drop non-JSON-serializable keys and history for a compact summary
    summary = []
    for r in results:
        s = {k: v for k, v in r.items() if k != "history"}
        summary.append(s)
    args_ser = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    out_json = args.out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump({"runs": summary, "args": args_ser}, f, indent=2)
    print(f"  saved {out_json}")

    # we save full histories for plotting
    plot_comparison(results, args.out_dir / "mnist_mmnn_vs_mlp.png")
    with open(args.out_dir / "histories.json", "w") as f:
        json.dump([{k: v for k, v in r.items()} for r in results], f, indent=2)

    # we update results.md Run 2 table from this run
    _update_results_md(args.out_dir, summary)


def _update_results_md(out_dir: Path, summary: list[dict]) -> None:
    """Overwrite the Run 2 section in results.md with the current summary table."""
    md_path = out_dir / "results.md"
    block = [
        "## Run 2 — MLP vs low-rank MMNN with random/frozen features (fixWb)",
        "",
        "| Model | Config | Params | Trainable | Test acc | Test loss |",
        "|-------|--------|--------|-----------|----------|-----------|",
    ]
    for r in summary:
        if r["arch"] == "MLP":
            model = "**MLP**"
            h, n = r.get("hidden", 512), r.get("num_hidden", 2)
            cfg = "784→" + "→".join([str(h)] * n) + "→10"
        else:
            R = r.get("ranks", [0, 0])[1] if len(r.get("ranks", [])) > 1 else "?"
            model = f"**MMNN R={R}**"
            cfg = "fixWb=True (random features)" if r.get("fix_wb") else "fixWb=False"
        block.append(f"| {model} | {cfg} | {r['nparams']:,} | {r.get('ntrainable', r['nparams']):,} | **{100*r['final_test_acc']:.2f}%** | {r['final_test_loss']:.4f} |")
    block.append("")
    replacement = "\n".join(block)
    if md_path.exists():
        raw = md_path.read_text()
        if "## Run 2" in raw:
            pre = raw.split("## Run 2")[0].rstrip()
            new_raw = pre + "\n\n" + replacement
        else:
            new_raw = raw.rstrip() + "\n\n" + replacement
        md_path.write_text(new_raw)
        print(f"  updated {md_path}")
    else:
        md_path.write_text("# MNIST: MLP vs MMNN\n\n" + replacement + "\n")
        print(f"  wrote {md_path}")


if __name__ == "__main__":
    main()

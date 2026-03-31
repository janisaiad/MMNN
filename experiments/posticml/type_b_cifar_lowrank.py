#!/usr/bin/env python3
"""
Type B (post-ICML): CIFAR-10 with low-rank MMNN-style stack (fixWb=True), width M=1024.

Flattened 32x32x3 -> alternating rank/width blocks -> 10 logits. SGD + momentum.

Usage:
  python experiments/posticml/type_b_cifar_lowrank.py --quick
  python experiments/posticml/type_b_cifar_lowrank.py --epochs 100 --hidden-rank 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from experiments.table.mmnn_vs import MMNN  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "results" / "type_b_cifar"
M_FIXED = 1024
CIFAR_DIM = 32 * 32 * 3


def load_cifar(train: bool, data_root: Path):
    try:
        import torchvision
        from torchvision import transforms
    except ImportError as e:
        raise SystemExit("Install torchvision for Type B (pip install torchvision)") from e

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    ds = torchvision.datasets.CIFAR10(
        root=str(data_root), train=train, download=True, transform=tfm
    )
    return ds


def train_epoch(model, loader, opt, device, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main() -> None:
    p = argparse.ArgumentParser(description="Type B: CIFAR-10 low-rank MMNN, M=1024")
    p.add_argument("--quick", action="store_true", help="2 epochs, small batch")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-rank", type=int, default=20)
    p.add_argument("--num-layers", type=int, default=3, help="number of ReLU blocks (rank->width->rank)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data" / "cifar10")
    p.add_argument("--overwrite", action="store_true", help="remove previous json/pth for this run name")
    args = p.parse_args()

    epochs = 2 if args.quick else (args.epochs or 50)
    batch_size = 256 if args.quick else args.batch_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = load_cifar(True, args.data_root)
    test_ds = load_cifar(False, args.data_root)
    nw = 0  # avoid worker issues on shared clusters
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())

    num_layers = args.num_layers
    r = args.hidden_rank
    ranks = [CIFAR_DIM] + [r] * num_layers + [10]
    widths = [M_FIXED] * (num_layers + 1)
    model = MMNN(ranks=ranks, widths=widths, device=str(device), ResNet=False, fixWb=True).to(device)

    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"cifar10_M{M_FIXED}_r{r}_L{num_layers}_e{epochs}.json"
    pth_path = OUT_ROOT / f"cifar10_M{M_FIXED}_r{r}_L{num_layers}_e{epochs}.pth"
    if args.overwrite:
        for pth in (out_path, pth_path):
            if pth.exists():
                pth.unlink()
    history = []
    best_acc = 0.0
    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, device, criterion)
        te_loss, te_acc = eval_epoch(model, test_loader, device, criterion)
        sched.step()
        best_acc = max(best_acc, te_acc)
        history.append({"epoch": ep + 1, "train_loss": tr_loss, "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc})
        print(f"epoch {ep+1}/{epochs} train_loss={tr_loss:.4f} acc={tr_acc:.4f} test_acc={te_acc:.4f}")

    out = {
        "M_width": M_FIXED,
        "hidden_rank": r,
        "num_layers": num_layers,
        "epochs": epochs,
        "best_test_acc": best_acc,
        "final_test_acc": history[-1]["test_acc"] if history else None,
        "history": history,
        "params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    torch.save(model.state_dict(), OUT_ROOT / f"cifar10_M{M_FIXED}_r{r}_L{num_layers}_e{epochs}.pth")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
we train low-bottleneck attention encoders on sum-of-cosines 1D regression (MMNN-style target).
we save config.json, losses.json, result.json, best_model.pt, optional checkpoints, plots, token_partials.npz.
usage:
  python experiments/attention/attention.py --quick
  python experiments/attention/attention.py --depth-max 2 --heads 4 --residual-only both
  python experiments/attention/attention.py --factor-list 3,4,5 --heads-list 1,2 --d-k 5 --rank-ff 5
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_REPO = Path(__file__).resolve().parent.parent.parent


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_x_scale_sweep(spec: str) -> list[float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError("x_scale_sweep must be min:step:max")
    lo, step, hi = float(parts[0]), float(parts[1]), float(parts[2])
    if step <= 0:
        raise ValueError("x_scale_sweep step must be positive")
    out: list[float] = []
    x = lo
    while x <= hi + 1e-9:
        out.append(round(x, 10))
        x += step
    return out


def target_sum_cos(x: torch.Tensor, factor: int, x_scale: float) -> torch.Tensor:
    """we compute sum_{k=1}^{factor} cos(2 pi k x_scale x) on [-1,1]"""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    device, dtype = x.device, x.dtype
    ks = torch.arange(1, factor + 1, device=device, dtype=dtype).view(1, -1)
    xx = x.view(-1, 1) * x_scale
    return torch.cos(2 * math.pi * ks * xx).sum(dim=1, keepdim=True)


class AttentionBlockNoResidual(nn.Module):
    """we apply norm -> MHA -> norm -> FFN without x + sublayer(x) (no residual)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm2(a)
        return self.ff(h)


class AttentionBlockWithResidual(nn.Module):
    """we wrap PyTorch TransformerEncoderLayer (Pre-LN + residuals)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="relu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class LowRankAttentionRegressor(nn.Module):
    """we map scalar x -> (B, seq_len, d_model), stack encoder blocks, mean-pool -> scalar y."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        num_heads: int,
        d_k: int,
        rank_ff: int,
        depth: int,
        use_residual: bool,
    ) -> None:
        super().__init__()
        if d_model != num_heads * d_k:
            raise ValueError("d_model must be num_heads * d_k")
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        if use_residual:
            self.blocks = nn.ModuleList(
                [AttentionBlockWithResidual(d_model, num_heads, rank_ff) for _ in range(depth)]
            )
        else:
            self.blocks = nn.ModuleList(
                [AttentionBlockNoResidual(d_model, num_heads, rank_ff) for _ in range(depth)]
            )
        self.head = nn.Linear(d_model, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x).unsqueeze(1).expand(-1, self.seq_len, -1)
        h = h + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_tokens(x)
        return self.head(h.mean(dim=1))


def smooth_tail(losses: Sequence[float], w: int) -> float:
    if len(losses) == 0:
        return float("nan")
    tail = losses[-w:] if len(losses) >= w else losses
    return float(sum(tail) / len(tail))


def train_one(
    *,
    out_root: Path,
    depth: int,
    num_heads: int,
    d_k: int,
    rank_ff: int,
    seq_len: int,
    factor: int,
    x_scale: float,
    use_residual: bool,
    n_train: int,
    batch_size: int,
    lr: float,
    max_epochs: int,
    grad_clip: float,
    early_patience: int,
    early_min_delta: float,
    smooth_window: int,
    save_every: int,
    lr_floor: float | None,
    lr_schedule: str | None,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    d_model = num_heads * d_k
    model = LowRankAttentionRegressor(
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        d_k=d_k,
        rank_ff=rank_ff,
        depth=depth,
        use_residual=use_residual,
    ).to(device)

    def run_name() -> str:
        parts = [f"d{depth}", f"h{num_heads}", f"dk{d_k}", f"ff{rank_ff}"]
        if factor != 3:
            parts.append(f"f{factor}")
        if abs(x_scale - 1.0) > 1e-12:
            xs = str(x_scale).replace(".", "p")
            parts.append(f"xs{xs}")
        parts.append("res" if use_residual else "nores")
        return "_".join(parts)

    name = run_name()
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    interval = (-1.0, 1.0)
    x_train = np.linspace(interval[0], interval[1], n_train, dtype=np.float64)
    y_train_np = np.zeros(n_train, dtype=np.float64)
    xt = torch.tensor(x_train, device=device, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        yt = target_sum_cos(xt, factor, x_scale)
        y_train_np = yt.cpu().numpy().ravel()

    n_test = min(2048, max(256, n_train // 2))
    x_test = np.linspace(interval[0], interval[1], n_test, dtype=np.float64)
    x_test_t = torch.tensor(x_test, device=device, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        y_test_t = target_sum_cos(x_test_t, factor, x_scale)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    loss_fn = nn.MSELoss()

    all_losses: list[float] = []
    best_smooth = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_no_improve = 0
    current_lr = lr
    stopped_reason = "max_epochs"
    ckpt_dir = out_dir / "checkpoints"
    if save_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"we run {name} on {device} ...", flush=True)
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_losses: list[float] = []
        for s in range(0, n_train, batch_size):
            idx = perm[s : s + batch_size]
            xb = xt[idx]
            yb = torch.tensor(y_train_np, device=device, dtype=torch.float32).view(-1, 1)[idx]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_mse = float(np.mean(epoch_losses))
        all_losses.append(train_mse)
        smooth = smooth_tail(all_losses, smooth_window)

        if smooth + early_min_delta < best_smooth:
            best_smooth = smooth
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        halve = lr_schedule == "halve_on_plateau_no_early_stop"
        if halve and epochs_no_improve >= early_patience:
            new_lr = current_lr * 0.5
            if lr_floor is not None and new_lr < lr_floor:
                stopped_reason = "lr_floor"
                print(f"    final_lr={current_lr:.4e} (next would be below lr_floor)", flush=True)
                break
            print(f"    lr_decay epoch={epoch} new_lr={new_lr:.4e} (plateau {early_patience} epochs)", flush=True)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
            current_lr = new_lr
            epochs_no_improve = 0

        if save_every > 0 and epoch % save_every == 0 and best_state is not None:
            torch.save(best_state, ckpt_dir / f"model_epoch_{epoch:05d}.pt")

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"    epoch {epoch}/{max_epochs} train_mse={train_mse:.4e} smooth={smooth:.4e} "
                f"no_improve={epochs_no_improve} lr={current_lr:.4e}",
                flush=True,
            )

        if not halve and epochs_no_improve >= early_patience:
            stopped_reason = "early_plateau"
            break

    elapsed = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    torch.save(model.state_dict(), out_dir / "final_model.pt")

    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t)
        test_mse = float(loss_fn(pred_test, y_test_t).item())
        final_train_mse = float(all_losses[-1]) if all_losses else float("nan")

    config_payload: dict[str, Any] = {
        "depth": depth,
        "num_heads": num_heads,
        "d_k": d_k,
        "rank_ff": rank_ff,
        "seq_len": seq_len,
        "d_model": d_model,
        "use_residual": use_residual,
        "factor": factor,
        "x_scale": x_scale,
        "n_train": n_train,
        "batch_size": batch_size,
        "lr": lr,
        "max_epochs": max_epochs,
        "grad_clip": grad_clip,
        "early_patience": early_patience,
        "early_min_delta": early_min_delta,
        "smooth_window": smooth_window,
        "save_every": save_every,
        "optimizer": "sgd",
        "momentum": 0.0,
        "target": "sum_cos_2pi_k",
        "interval": list(interval),
        "seed": seed,
    }
    if lr_floor is not None:
        config_payload["lr_floor"] = lr_floor
    if lr_schedule:
        config_payload["lr_schedule"] = lr_schedule

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    losses_payload = {"all_losses": all_losses, "losses": all_losses}
    with open(out_dir / "losses.json", "w", encoding="utf-8") as f:
        json.dump(losses_payload, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(range(1, len(all_losses) + 1), all_losses, lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train MSE")
    ax.set_title(f"{name} (train)")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_loss.png", dpi=140)
    plt.close(fig)

    x_plot = np.linspace(interval[0], interval[1], 400, dtype=np.float64)
    x_plot_t = torch.tensor(x_plot, device=device, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        y_plot_pred = model(x_plot_t).cpu().numpy().ravel()
    y_plot_true = np.zeros_like(x_plot)
    with torch.no_grad():
        y_plot_true = target_sum_cos(torch.tensor(x_plot, device=device).view(-1, 1), factor, x_scale).cpu().numpy().ravel()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_plot, y_plot_true, label="target", lw=2)
    ax.plot(x_plot, y_plot_pred, label="model", lw=1.5, alpha=0.85)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"fit (factor={factor}, x_scale={x_scale})")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_fit.png", dpi=140)
    plt.close(fig)

    with torch.no_grad():
        tok = model.forward_tokens(x_plot_t[:200])
        dim0 = tok[:, :, 0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    for j in range(min(8, dim0.shape[1])):
        ax.plot(x_plot[:200], dim0[:, j], lw=0.8, alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("token dim0 (subset of seq)")
    ax.set_title("token partials (dim 0, first seq positions)")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_token_partials_dim0.png", dpi=140)
    plt.close(fig)

    np.savez_compressed(
        out_dir / "token_partials.npz",
        x=x_plot[:200],
        tokens=tok.cpu().numpy()[:200],
        prediction=y_plot_pred[:200],
        target=y_plot_true[:200],
    )

    n_params = sum(p.numel() for p in model.parameters())
    early_stopped = stopped_reason != "max_epochs"

    artifacts = [
        "config.json",
        "result.json",
        "losses.json",
        "best_model.pt",
        "final_model.pt",
        "plot_loss.png",
        "plot_fit.png",
        "plot_token_partials_dim0.png",
        "token_partials.npz",
    ]
    if save_every > 0:
        artifacts.append("checkpoints/")

    row: dict[str, Any] = {
        **config_payload,
        "epochs_run": len(all_losses),
        "final_train_mse": final_train_mse,
        "best_smooth_train_mse": best_smooth,
        "test_mse": test_mse,
        "seconds": elapsed,
        "n_params": n_params,
        "early_stopped": early_stopped,
        "stopped_reason": stopped_reason,
        "artifacts": artifacts,
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print(
        f"  done epochs={len(all_losses)} train_mse~{final_train_mse:.4e} test_mse={test_mse:.4e} "
        f"early={early_stopped} ({stopped_reason})",
        flush=True,
    )
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=str, default=str(_REPO / "experiments" / "attention" / "results_attention"))
    p.add_argument("--factor", type=int, default=3, help="harmonics 1..factor in sum cos")
    p.add_argument("--factor-list", type=str, default="", help="comma factors; overrides single --factor when set")
    p.add_argument("--x-scale", type=float, default=1.0)
    p.add_argument("--x-scale-sweep", type=str, default="", help="min:step:max; overrides --x-scale, one run per value")
    p.add_argument("--d-k", type=int, default=10)
    p.add_argument("--rank-ff", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max-epochs", type=int, default=3000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-patience", type=int, default=50)
    p.add_argument("--early-min-delta", type=float, default=1e-5)
    p.add_argument("--lr-floor", type=float, default=1e-6)
    p.add_argument(
        "--lr-schedule",
        type=str,
        default="",
        choices=["", "halve_on_plateau_no_early_stop"],
        help="we halve lr on plateau until next lr would be below --lr-floor",
    )
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--n-train", type=int, default=768)
    p.add_argument("--depth-min", type=int, default=1)
    p.add_argument("--depth-max", type=int, default=10)
    p.add_argument("--depth-list", type=str, default="", help="comma depths; overrides depth-min/max when set")
    p.add_argument("--heads", type=str, default="1,2,4,6,8,12,16")
    p.add_argument("--heads-list", type=str, default="", help="comma heads; overrides --heads when set")
    p.add_argument(
        "--residual-only",
        type=str,
        default="both",
        choices=["both", "yes", "no"],
        help="both=res+nores per config, yes=only residual, no=only no-residual",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_train = 256
        max_epochs = 200
        seq_len = 16
        save_every = max(50, args.save_every)
        heads_list = [4]
        depth_list = [2]
        factor_list = [3]
        x_scales = [float(args.x_scale)]
        lr_sched = ""
        lr_floor = None
    else:
        n_train = args.n_train
        max_epochs = args.max_epochs
        seq_len = args.seq_len
        save_every = args.save_every
        heads_list = parse_int_list(args.heads_list) if args.heads_list.strip() else parse_int_list(args.heads)
        if args.depth_list.strip():
            depth_list = parse_int_list(args.depth_list)
        else:
            depth_list = list(range(args.depth_min, args.depth_max + 1))
        if args.factor_list.strip():
            factor_list = parse_int_list(args.factor_list)
        else:
            factor_list = [args.factor]
        if args.x_scale_sweep.strip():
            x_scales = parse_x_scale_sweep(args.x_scale_sweep)
        else:
            x_scales = [float(args.x_scale)]
        lr_sched = args.lr_schedule or None
        lr_floor = float(args.lr_floor) if lr_sched == "halve_on_plateau_no_early_stop" else None

    if args.residual_only == "both":
        residual_modes = [True, False]
    elif args.residual_only == "yes":
        residual_modes = [True]
    else:
        residual_modes = [False]

    summary_path = out_root / "summary.json"
    summary: list[dict[str, Any]] = []
    if summary_path.is_file():
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
        except json.JSONDecodeError:
            summary = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for x_scale in x_scales:
        for depth in depth_list:
            for factor in factor_list:
                for num_heads in heads_list:
                    d_k = int(args.d_k)
                    rank_ff = int(args.rank_ff)
                    d_model = num_heads * d_k
                    if d_model % num_heads != 0:
                        print(f"skip d_model not divisible by heads: h={num_heads} dk={d_k}", flush=True)
                        continue
                    for use_residual in residual_modes:
                        row = train_one(
                            out_root=out_root,
                            depth=depth,
                            num_heads=num_heads,
                            d_k=d_k,
                            rank_ff=rank_ff,
                            seq_len=seq_len,
                            factor=factor,
                            x_scale=x_scale,
                            use_residual=use_residual,
                            n_train=n_train,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            max_epochs=max_epochs,
                            grad_clip=args.grad_clip,
                            early_patience=args.early_patience,
                            early_min_delta=args.early_min_delta,
                            smooth_window=args.smooth_window,
                            save_every=save_every,
                            lr_floor=lr_floor,
                            lr_schedule=lr_sched,
                            seed=args.seed,
                            device=device,
                        )
                        parts = [f"d{depth}", f"h{num_heads}", f"dk{d_k}", f"ff{rank_ff}"]
                        if factor != 3:
                            parts.append(f"f{factor}")
                        if abs(x_scale - 1.0) > 1e-12:
                            parts.append(f"xs{str(x_scale).replace('.', 'p')}")
                        parts.append("res" if row["use_residual"] else "nores")
                        run_id = "_".join(parts)
                        row["run_id"] = run_id
                        summary = [r for r in summary if r.get("run_id") != run_id]
                        summary.append(row)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

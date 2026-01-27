#!/usr/bin/env python3
"""
Track log ratios R_{i,j} = log|f_i| - log|f_j| during training at x=0.

Uses the same config as factor=4, rank=15 (15 partial functions, 225 pairs).
At each checkpoint we compute f_k and R at x=0, save all trajectories,
and plot the max-ratio vs epoch and vs time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# we add project root to path (logratio -> table -> experiments -> MMNN)
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from experiments.table.mmnn_vs import MMNN

# we configure matplotlib for beautiful LaTeX-style figures
plt.rcParams["figure.figsize"] = [12, 10]
plt.rcParams["font.size"] = 18
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.rm"] = "serif"
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.size"] = 22
mpl.rcParams["axes.formatter.limits"] = (-6, 6)
mpl.rcParams["axes.formatter.use_mathtext"] = True
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.top"] = True


def target_function(x: np.ndarray, factor: int) -> np.ndarray:
    """cos(2*factor*pi*x) + cos(2*pi*x)."""
    return np.cos(2 * factor * np.pi * x) + np.cos(2 * np.pi * x)


def get_partial_fk_layer2(
    model: nn.Module,
    x_loc: float,
    device: torch.device,
    dtype: torch.dtype,
    eps_x: float = 1e-8,
) -> np.ndarray:
    """
    Compute layer-2 partial functions f_k at x = x_loc.

    Layer 2 = fcs[0]->ReLU->fcs[1]->fcs[2]->ReLU->fcs[3], output [batch, rank].
    Returns f_k of shape (r,) with r = hidden_rank.
    """
    if abs(x_loc) < 1e-12:
        x_loc = eps_x
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor([[x_loc]], device=device, dtype=dtype)
        h = model.fcs[0](x_t)
        h = torch.relu(h)
        h = model.fcs[1](h)
        h = model.fcs[2](h)
        h = torch.relu(h)
        h = model.fcs[3](h)
        f_k = h.cpu().numpy().flatten()
    return f_k


def compute_log_ratio_matrix(
    f_k: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    R[i,j] = log(|f_i| + eps) - log(|f_j| + eps).
    Shape (r, r).
    """
    r = len(f_k)
    log_f = np.log(np.abs(f_k) + eps)
    R = np.zeros((r, r))
    for i in range(r):
        for j in range(r):
            R[i, j] = log_f[i] - log_f[j]
    return R


def run_training_with_logratio_tracking(
    config: dict,
    output_dir: Path,
    checkpoint_every: int = 50,
    x_loc: float = 0.0,
    eps: float = 1e-6,
    seed: int | None = 42,
    save_plot: bool = True,
) -> None:
    """
    Train with config, track f_k and R at x=x_loc every checkpoint_every epochs.
    Save all 225 trajectories and plot max-ratio vs epoch / time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    factor = int(config["factor"])
    hidden_rank = int(config["hidden_rank"])
    hidden_width = int(config.get("hidden_width", 1024))
    num_layers = int(config.get("num_layers", 2))
    num_epochs = int(config.get("num_epochs", 10000))
    lr_init = float(config.get("lr_init", 0.01))
    optimizer_type = config.get("optimizer_type", "SGD")
    scheduler_type = config.get("scheduler_type")
    scheduler_params = config.get("scheduler_params", {})
    momentum = float(config.get("momentum", 0.3))
    batch_size = max(1, int(config.get("batch_size", 4 * factor * 10)))

    ranks = [1] + [hidden_rank] * num_layers + [1]
    widths = [hidden_width] * (num_layers + 1)
    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=device,
        ResNet=False,
        fixWb=True,
    )

    interval = [-1.0, 1.0]
    n_train = max(1, int(factor * hidden_width))
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_function(x_train, factor)
    x_train_t = torch.tensor(x_train.reshape(-1, 1), device=device, dtype=dtype)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), device=device, dtype=dtype)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)
    else:
        betas = tuple(config.get("betas", (0.9, 0.999)))
        optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=betas)

    adaptive_scheduler = None
    if scheduler_type == "AdaptiveStagnation":
        adaptive_scheduler = {
            "lr_sequence": scheduler_params.get(
                "lr_sequence",
                [0.01, 0.005, 0.001, 0.0005, 0.0001],
            ),
            "current_lr_index": 0,
            "window_size": int(scheduler_params.get("window_size", 10)),
            "min_epochs_before_reduce": int(
                scheduler_params.get("min_epochs_before_reduce", 20)
            ),
            "last_reduction_epoch": -1,
        }

    use_adam_first = optimizer_type == "Adam"
    switched_to_sgd = False
    sgd_momentum = float(config.get("momentum", 0.3))

    epochs_list: list[int] = []
    times_list: list[float] = []
    fk_list: list[np.ndarray] = []
    R_list: list[np.ndarray] = []

    start_time = time.perf_counter()
    all_losses: list[float] = []

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    pbar = tqdm(
        range(num_epochs),
        desc="train",
        unit="epoch",
    )

    for epoch in pbar:
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            x_b = x_train_t[idx]
            y_b = y_train_t[idx]
            optimizer.zero_grad()
            pred = model(x_b)
            loss = nn.MSELoss()(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches
        all_losses.append(epoch_loss)

        # switch Adam -> SGD when loss < 1e-3
        if use_adam_first and not switched_to_sgd and epoch_loss < 1e-3:
            lr = optimizer.param_groups[0]["lr"]
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=sgd_momentum)
            switched_to_sgd = True

        # adaptive stagnation LR reduction
        if adaptive_scheduler is not None:
            ci = adaptive_scheduler["current_lr_index"]
            seq = adaptive_scheduler["lr_sequence"]
            win = adaptive_scheduler["window_size"]
            min_ep = adaptive_scheduler["min_epochs_before_reduce"]
            last_red = adaptive_scheduler["last_reduction_epoch"]
            if (
                epoch >= min_ep
                and epoch - last_red >= min_ep
                and len(all_losses) >= 2 * win
                and ci < len(seq) - 1
            ):
                recent = np.mean(all_losses[-win:])
                prev = np.mean(all_losses[-2 * win : -win])
                if recent >= prev:
                    ci += 1
                    new_lr = seq[ci]
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    adaptive_scheduler["current_lr_index"] = ci
                    adaptive_scheduler["last_reduction_epoch"] = epoch

        # checkpoint: compute f_k and R at x=x_loc
        if epoch % checkpoint_every == 0 or epoch == 0:
            t_now = time.perf_counter() - start_time
            f_k = get_partial_fk_layer2(model, x_loc, device, dtype)
            R = compute_log_ratio_matrix(f_k, eps=eps)
            epochs_list.append(epoch)
            times_list.append(t_now)
            fk_list.append(f_k.copy())
            R_list.append(R.copy())

        pbar.set_postfix({"loss": f"{epoch_loss:.4e}"})

    # save arrays
    epochs_arr = np.array(epochs_list, dtype=np.int64)
    times_arr = np.array(times_list, dtype=np.float64)
    fk_arr = np.stack(fk_list, axis=0)  # (n_check, 15)
    R_arr = np.stack(R_list, axis=0)    # (n_check, 15, 15)

    np.save(output_dir / "epochs.npy", epochs_arr)
    np.save(output_dir / "times.npy", times_arr)
    np.save(output_dir / "fk_x0.npy", fk_arr)
    np.save(output_dir / "R_x0.npy", R_arr)

    # flatten to 225 trajectories: row-major (i,j) -> idx = i*15 + j
    n_check = R_arr.shape[0]
    trajectories_225 = R_arr.reshape(n_check, -1)  # (n_check, 225)
    np.save(output_dir / "trajectories_225.npy", trajectories_225)

    # max ratio over (i,j) at each checkpoint
    max_ratio = np.max(R_arr, axis=(1, 2))
    np.save(output_dir / "max_ratio.npy", max_ratio)

    # plot: max ratio vs epoch only (no wall-clock time)
    if save_plot:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        ax.plot(epochs_arr, max_ratio, "b-", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\max_{i,j} R_{i,j}(x=0)$")
        ax.set_title(r"Trajectory of $\max_{i,j} R_{i,j}(x=0)$ vs epoch")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "trajectory_max_ratio.png", bbox_inches="tight")
        plt.close()

    # save a short summary
    summary = {
        "n_checkpoints": int(n_check),
        "checkpoint_every": checkpoint_every,
        "x_loc": float(x_loc),
        "eps": float(eps),
        "rank": int(hidden_rank),
        "n_pairs": 225,
        "max_ratio_final": float(max_ratio[-1]) if len(max_ratio) else None,
        "max_ratio_max": float(np.max(max_ratio)),
        "training_time_s": float(times_arr[-1]) if len(times_arr) else None,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved all data to {output_dir}")
    print(f"  epochs.npy, times.npy, fk_x0.npy, R_x0.npy, trajectories_225.npy, max_ratio.npy")
    if save_plot:
        print(f"  trajectory_max_ratio.png")
    print(f"  summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Track log ratios during training (factor=4, rank=15, x=0)."
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.json (default: use factor4 rank15 SGD mom0.3 AdaptiveStagnation).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: logratio/factor4_rank15_SGD_mom0.3_...)",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Checkpoint every N epochs (default: 50).",
    )
    ap.add_argument(
        "--x",
        type=float,
        default=0.0,
        help="Input location for partials (default: 0.0).",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Epsilon for log ratios (default: 1e-6).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed (used if --seeds not set).")
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help='Space-separated seeds for 3 runs, e.g. "42 43 44". Produces one plot with 3 curves vs epoch (no wall-clock).',
    )
    ap.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override num_epochs from config (default: use config).",
    )
    args = ap.parse_args()

    if args.config is not None and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            "factor": 4,
            "num_layers": 2,
            "hidden_rank": 15,
            "hidden_width": 1024,
            "num_epochs": 10000,
            "batch_size": 160,
            "lr_init": 0.01,
            "optimizer_type": "SGD",
            "scheduler_type": "AdaptiveStagnation",
            "scheduler_params": {
                "lr_sequence": [0.01, 0.005, 0.001, 0.0005, 0.0001],
                "window_size": 10,
                "min_epochs_before_reduce": 20,
            },
            "momentum": 0.3,
            "parameterization": "NTK",
        }

    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs

    base = Path(__file__).resolve().parent
    cfg_name = (
        f"factor{config['factor']}_rank{config['hidden_rank']}"
        f"_SGD_mom{config.get('momentum', 0.3)}_lr{config['lr_init']}_AdaptiveStagnation"
    )
    out = args.out_dir
    if out is None:
        out = base / "runs" / cfg_name

    seeds_raw = args.seeds
    if seeds_raw is not None:
        seeds = [int(s) for s in seeds_raw.split()]
        if len(seeds) < 1:
            raise SystemExit("--seeds must list at least one seed, e.g. \"42 43 44\".")
        out_base = out / "3seeds" if len(seeds) > 1 else out
        out_base.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            run_dir = out_base / f"seed_{s}"
            run_training_with_logratio_tracking(
                config=config,
                output_dir=run_dir,
                checkpoint_every=args.checkpoint_every,
                x_loc=args.x,
                eps=args.eps,
                seed=s,
                save_plot=False,
            )
        if len(seeds) >= 2:
            # one plot: 3 curves vs epoch (no wall-clock)
            ep = np.load(out_base / f"seed_{seeds[0]}" / "epochs.npy")
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"][: len(seeds)]
            fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
            for i, s in enumerate(seeds):
                mr = np.load(out_base / f"seed_{s}" / "max_ratio.npy")
                ax.plot(ep, mr, color=colors[i % len(colors)], linewidth=1.5, alpha=0.8, label=f"seed {s}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(r"$\max_{i,j} R_{i,j}(x=0)$")
            ax.set_title(r"Trajectory of $\max_{i,j} R_{i,j}(x=0)$ vs epoch")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = out_base / "trajectory_max_ratio.png"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"Saved {plot_path} (3 curves vs epoch, no wall-clock).")
        else:
            # one seed with --seeds "42": run already done, create single plot from saved data
            run_dir = out_base / f"seed_{seeds[0]}"
            ep = np.load(run_dir / "epochs.npy")
            mr = np.load(run_dir / "max_ratio.npy")
            fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
            ax.plot(ep, mr, "b-", linewidth=1.5, alpha=0.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(r"$\max_{i,j} R_{i,j}(x=0)$")
            ax.set_title(r"Trajectory of $\max_{i,j} R_{i,j}(x=0)$ vs epoch")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = out_base / "trajectory_max_ratio.png"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"Saved {plot_path}.")
        return

    run_training_with_logratio_tracking(
        config=config,
        output_dir=out,
        checkpoint_every=args.checkpoint_every,
        x_loc=args.x,
        eps=args.eps,
        seed=args.seed,
        save_plot=True,
    )


if __name__ == "__main__":
    main()

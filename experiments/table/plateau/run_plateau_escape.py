#!/usr/bin/env python3
"""
we run plateau-escape experiments: same config as a long GD run (e.g. f3_N768_L3),
varying lr and batch size; train until loss < threshold (plateau escape) or max_epochs.
we measure (1) number of epochs to escape, (2) L2 norm of (params_at_escape - params_at_init).
we also track log-ratio distribution at x=0 (r*(r-1)/2 pairs) with LR-dependent checkpoint interval,
save logratio_epochs.npy and logratio_values.npy per run, then make a GIF of the distribution and plot trajectories.
output: experiments/table/plateau/; at end plot time-to-escape vs lr (log-log) and histogram of norms.
usage: python run_plateau_escape.py [--config PATH] [--threshold 1.2e-2] [--max-epochs N]
  stop training when plateau escaped (loss < threshold).
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

# we add repo root to path so that "from experiments.table ..." resolves (plateau -> table -> experiments -> MMNN)
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from experiments.table.mmnn_vs import MMNN
from experiments.table.run_scaling_law_depth_width import SWEEP_RANK, SWEEP_WIDTH, target_baseline

_BASE = Path(__file__).resolve().parent
PLATEAU_DIR = _BASE
DEFAULT_CONFIG_PATH = _REPO / "experiments" / "table" / "results_sumcos_selected_rerun_lowlr" / "f3_N768_bs128_L3" / "config.json"

# we run with decreasing lr (largest first): 1e-2 down to 1e-4; stop at epoch 0.7/lr per run
LR_LIST = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
# we run with batch size 4 only; escape threshold 1e-1
BATCH_SIZES = [4]
ESCAPE_THRESHOLD = 1e-1
MAX_EPOCHS = 500_000
# we stop each run at epoch 4/lr (capped by MAX_EPOCHS)
MAX_EPOCHS_PER_LR_FACTOR = 4.0
SEED = 42
X_LOC = 0.0
EPS_LOG = 1e-6
# we cap log-ratio checkpoints so long runs stay fast (was causing ~2 s/epoch when checkpointing every 5)
MAX_LOGratio_CHECKPOINTS = 500

# we store log-ratio distribution every N epochs (2x more checkpoints than before)
LR_TO_CHECKPOINT_EVERY: dict[float, int] = {
    1e-4: 50,
    2e-4: 25,
    5e-4: 12,
    1e-3: 10,
    2e-3: 12,
    5e-3: 25,
    1e-2: 5,
}


def checkpoint_every_for_lr(lr: float) -> int:
    """return epoch interval for saving log-ratio distribution for this lr (2x more checkpoints)."""
    if lr in LR_TO_CHECKPOINT_EVERY:
        return LR_TO_CHECKPOINT_EVERY[lr]
    if lr <= 1e-4:
        return 50
    if lr <= 2e-4:
        return 25
    if lr <= 5e-4:
        return 12
    if lr <= 1e-3:
        return 10
    if lr <= 2e-3:
        return 12
    if lr <= 5e-3:
        return 25
    return 5


def get_partial_fk_at_block(
    model: nn.Module,
    x_loc: float,
    device: torch.device,
    dtype: torch.dtype,
    block_index: int,
    eps_x: float = 1e-8,
) -> np.ndarray:
    """
    we compute bottleneck activations f_k at x = x_loc after the full block block_index (0-indexed).
    block 0 = after fcs[1], block 1 = after fcs[3], etc. returns f_k of shape (r,) with r = hidden_rank.
    """
    if abs(x_loc) < 1e-12:
        x_loc = eps_x
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor([[x_loc]], device=device, dtype=dtype)
        depth = getattr(model, "depth", None)
        if depth is None:
            depth = (len(model.fcs) + 1) // 2
        n_blocks = depth - 1
        if block_index < 0 or block_index >= n_blocks:
            raise ValueError(f"block_index must be in [0, {n_blocks}), got {block_index}")
        for j in range(block_index + 1):
            x_t = model.fcs[2 * j](x_t)
            x_t = torch.relu(x_t)
            x_t = model.fcs[2 * j + 1](x_t)
        f_k = x_t.cpu().numpy().flatten()
    return f_k


def get_partial_fk_last_hidden(
    model: nn.Module,
    x_loc: float,
    device: torch.device,
    dtype: torch.dtype,
    eps_x: float = 1e-8,
) -> np.ndarray:
    """
    we compute last-hidden-layer partials f_k at x = x_loc (low-rank channels before output).
    returns f_k of shape (r,) with r = hidden_rank.
    """
    depth = getattr(model, "depth", None)
    if depth is None:
        depth = (len(model.fcs) + 1) // 2
    return get_partial_fk_at_block(model, x_loc, device, dtype, block_index=depth - 2, eps_x=eps_x)


def compute_log_ratio_pairs(f_k: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    we compute r*(r-1)/2 pairwise log-ratios R_ij = log(|f_i|+eps) - log(|f_j|+eps) for i < j.
    returns shape (r*(r-1)//2,) in order (0,1),(0,2),...,(0,r-1),(1,2),...,(r-2,r-1).
    """
    r = len(f_k)
    log_f = np.log(np.abs(f_k) + eps)
    pairs: list[float] = []
    for i in range(r):
        for j in range(i + 1, r):
            pairs.append(float(log_f[i] - log_f[j]))
    return np.array(pairs, dtype=np.float64)


def param_norm_diff(state_init: dict, state_final: dict, device: torch.device) -> float:
    """L2 norm of (params_final - params_init) over all parameters."""
    total_sq = 0.0
    for k in state_init:
        if k not in state_final:
            continue
        a = state_init[k].to(device, dtype=torch.float32)
        b = state_final[k].to(device, dtype=torch.float32)
        if a.shape != b.shape:
            continue
        total_sq += ((b - a) ** 2).sum().item()
    return math.sqrt(total_sq)


def run_one(
    base_config: dict,
    lr: float,
    batch_size: int,
    output_dir: Path,
    threshold: float,
    max_epochs: int,
    seed: int,
) -> dict:
    """train until loss < threshold or max_epochs; return epochs_to_escape, norm_diff, escaped, all_losses."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    factor = int(base_config.get("factor", 3))
    n_train = int(base_config.get("n_train", 768))
    num_layers = int(base_config.get("num_layers", 3))
    hidden_width = int(base_config.get("hidden_width", SWEEP_WIDTH))
    hidden_rank = int(base_config.get("hidden_rank", SWEEP_RANK))

    ranks = [1] + [hidden_rank] * num_layers + [1]
    widths = [hidden_width] * (num_layers + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)

    init_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    interval = [-1.0, 1.0]
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_baseline(x_train, factor)
    x_t = torch.tensor(x_train.reshape(-1, 1), device=device, dtype=dtype)
    y_t = torch.tensor(y_train.reshape(-1, 1), device=device, dtype=dtype)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    all_losses: list[float] = []
    logratio_checkpoint_every = checkpoint_every_for_lr(lr)
    logratio_epochs_list: list[int] = []
    logratio_values_list: list[np.ndarray] = []
    logratio_values_2nd_list: list[np.ndarray] = []
    has_2nd_layer = num_layers >= 2

    run_config = {
        **base_config,
        "batch_size": batch_size,
        "lr": lr,
        "escape_threshold": threshold,
        "max_epochs": max_epochs,
        "logratio_checkpoint_every": logratio_checkpoint_every,
        "x_loc": X_LOC,
        "eps_log": EPS_LOG,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    escaped = False
    escape_epoch = None
    pbar = tqdm(range(max_epochs), desc=output_dir.name, unit="ep")
    for epoch in pbar:
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            x_b = x_t[idx]
            y_b = y_t[idx]
            optimizer.zero_grad()
            pred = model(x_b)
            loss = nn.MSELoss()(pred, y_b)
            if torch.isnan(loss) or torch.isinf(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if n_batches == 0:
            break
        epoch_loss /= n_batches
        all_losses.append(float(epoch_loss))
        pbar.set_postfix(loss=f"{epoch_loss:.4e}")

        # we save log-ratio distribution at x=0 every logratio_checkpoint_every epochs (capped to avoid slowdown)
        if len(logratio_epochs_list) < MAX_LOGratio_CHECKPOINTS and (epoch % logratio_checkpoint_every == 0 or epoch == 0):
            f_k = get_partial_fk_last_hidden(model, X_LOC, device, dtype)
            pairs = compute_log_ratio_pairs(f_k, eps=EPS_LOG)
            logratio_epochs_list.append(epoch)
            logratio_values_list.append(pairs.copy())
            if has_2nd_layer:
                f_k_2nd = get_partial_fk_at_block(model, X_LOC, device, dtype, block_index=1)
                pairs_2nd = compute_log_ratio_pairs(f_k_2nd, eps=EPS_LOG)
                logratio_values_2nd_list.append(pairs_2nd.copy())
            # we save incrementally so interrupted runs (or early escape) still have log-ratio data
            n_check = len(logratio_epochs_list)
            if n_check == 1 or n_check % 5 == 0:
                epochs_arr = np.array(logratio_epochs_list, dtype=np.int64)
                values_arr = np.stack(logratio_values_list, axis=0)
                np.save(output_dir / "logratio_epochs.npy", epochs_arr)
                np.save(output_dir / "logratio_values.npy", values_arr)
                if logratio_values_2nd_list:
                    values_2nd_arr = np.stack(logratio_values_2nd_list, axis=0)
                    np.save(output_dir / "logratio_values_2nd.npy", values_2nd_arr)

        if epoch_loss < threshold:
            escaped = True
            escape_epoch = epoch + 1
            break

    # we save log-ratio trajectories and plot right away (so plots exist after each run, not only at end)
    run_name = output_dir.name
    if logratio_epochs_list:
        epochs_arr = np.array(logratio_epochs_list, dtype=np.int64)
        values_arr = np.stack(logratio_values_list, axis=0)
        np.save(output_dir / "logratio_epochs.npy", epochs_arr)
        np.save(output_dir / "logratio_values.npy", values_arr)
        if logratio_values_2nd_list:
            values_2nd_arr = np.stack(logratio_values_2nd_list, axis=0)
            np.save(output_dir / "logratio_values_2nd.npy", values_2nd_arr)
        try:
            plot_logratio_trajectories(output_dir, run_name, plot_filename="logratio_trajectories.png")
        except Exception as e:
            print(f"skip plot {run_name}: {e}")
        if logratio_values_2nd_list:
            try:
                plot_logratio_trajectories(
                    output_dir,
                    run_name,
                    layer_suffix=" (2nd layer)",
                    values_filename="logratio_values_2nd.npy",
                    plot_filename="logratio_trajectories_2nd.png",
                )
            except Exception as e:
                print(f"skip plot 2nd {run_name}: {e}")

    norm_diff = param_norm_diff(init_state, model.state_dict(), device) if escaped else float("nan")
    results = {
        "escaped": escaped,
        "epochs_to_escape": escape_epoch,
        "norm_diff": norm_diff,
        "final_loss": all_losses[-1] if all_losses else None,
        "n_epochs_run": len(all_losses),
        "all_losses": all_losses,
    }
    with open(output_dir / "results.json", "w") as f:
        out = {**results}
        if len(out.get("all_losses", [])) > 10000:
            out["all_losses"] = all_losses[:5000] + ["..."] + all_losses[-5000:]
        json.dump(out, f, indent=2)
    return results


def make_logratio_gif(
    run_dir: Path,
    run_name: str,
    nbins: int = 30,
    layer_suffix: str = "",
    values_filename: str = "logratio_values.npy",
    gif_filename: str = "logratio_distribution.gif",
) -> None:
    """
    we load logratio_epochs.npy and values_filename and build a GIF of the log-ratio distribution over time.
    layer_suffix is used in the title (e.g. " (last)" or " (2nd layer)"); gif_filename is the output file name.
    """
    epochs_path = run_dir / "logratio_epochs.npy"
    values_path = run_dir / values_filename
    gif_path = run_dir / gif_filename
    if gif_path.exists():
        return
    if not epochs_path.exists() or not values_path.exists():
        return
    if imageio is None:
        print("imageio not available; skipping log-ratio GIF")
        return
    try:
        epochs = np.load(epochs_path)
        values = np.load(values_path)
    except Exception as e:
        print(f"skip gif {run_dir.name}: load failed: {e}")
        return
    if epochs.size == 0 or values.size == 0:
        return
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
    mpl.rcParams["axes.formatter.use_mathtext"] = False
    mpl.rcParams["font.family"] = "serif"
    frames: list[np.ndarray] = []
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    title_prefix = f"{run_name}{layer_suffix}"
    for t in range(len(epochs)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.hist(values[t], bins=nbins, color="steelblue", alpha=0.8, edgecolor="white", range=(vmin, vmax))
        ax.set_xlabel("R_ij(x=0)  [log|fi| - log|fj|]")
        ax.set_ylabel("count")
        ax.set_title(f"{title_prefix}  epoch {int(epochs[t])}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
    imageio.mimsave(str(gif_path), frames, duration=0.15, loop=0)
    print("saved", gif_path)


def plot_logratio_trajectories(
    run_dir: Path,
    run_name: str,
    max_curves: int = 50,
    layer_suffix: str = "",
    values_filename: str = "logratio_values.npy",
    plot_filename: str = "logratio_trajectories.png",
) -> None:
    """
    we plot log-ratio trajectories (each of the r*(r-1)/2 pairs vs epoch); if too many we plot mean +/- std.
    layer_suffix is used in the title; values_filename and plot_filename select which layer and output file.
    """
    epochs_path = run_dir / "logratio_epochs.npy"
    values_path = run_dir / values_filename
    if not epochs_path.exists() or not values_path.exists():
        return
    try:
        epochs = np.load(epochs_path)
        values = np.load(values_path)
    except Exception as e:
        print(f"skip {run_dir.name}: load failed: {e}")
        return
    if epochs.size == 0 or values.size == 0:
        return
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    n_pairs = values.shape[1]
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
    mpl.rcParams["axes.formatter.limits"] = (-6, 6)
    mpl.rcParams["axes.formatter.use_mathtext"] = False
    mpl.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if n_pairs <= max_curves:
        for k in range(n_pairs):
            ax.plot(epochs, values[:, k], alpha=0.4, linewidth=0.8)
    else:
        mean_ = np.nanmean(values, axis=1)
        std_ = np.nanstd(values, axis=1)
        ax.fill_between(epochs, mean_ - std_, mean_ + std_, alpha=0.3, color="steelblue")
        ax.plot(epochs, mean_, color="steelblue", linewidth=1.5, label="mean +/- std")
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R_ij(x=0)  [log|fi| - log|fj|]")
    ax.set_title(f"Log-ratio trajectories  {run_name}{layer_suffix}  ({n_pairs} pairs)")
    ax.grid(True, alpha=0.3)
    try:
        plt.tight_layout()
        out_path = run_dir / plot_filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved", out_path)
    except Exception as e:
        plt.close(fig)
        print(f"skip {run_dir.name} {plot_filename}: plot failed: {e}")


def plot_loss_curve(run_dir: Path, run_name: str) -> None:
    """
    we plot loss vs epoch for this run; load from losses.npy or results.json all_losses.
    """
    losses_arr: np.ndarray | None = None
    if (run_dir / "losses.npy").exists():
        try:
            losses_arr = np.load(run_dir / "losses.npy")
        except Exception:
            pass
    if losses_arr is None or losses_arr.size == 0:
        res_path = run_dir / "results.json"
        if not res_path.exists():
            return
        try:
            with open(res_path) as f:
                res = json.load(f)
            all_losses = res.get("all_losses")
            if not all_losses or "..." in str(all_losses):
                return
            losses_arr = np.array([float(x) for x in all_losses if isinstance(x, (int, float))], dtype=np.float64)
        except Exception:
            return
    if losses_arr.size == 0:
        return
    epochs = np.arange(len(losses_arr), dtype=np.int64)
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
    mpl.rcParams["axes.formatter.use_mathtext"] = False
    mpl.rcParams["font.family"] = "serif"
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(epochs, losses_arr, "b-", linewidth=0.8, alpha=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"Loss curve  {run_name}")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        plt.tight_layout()
        fig.savefig(run_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved", run_dir / "loss_curve.png")
    except Exception as e:
        print(f"skip loss_curve {run_name}: {e}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Plateau escape: train until loss < threshold, measure epochs and param norm diff.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="base config JSON (factor, n_train, num_layers, etc.)")
    ap.add_argument("--threshold", type=float, default=ESCAPE_THRESHOLD, help="escape when loss < this (default 1.2e-2)")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="max epochs per run")
    ap.add_argument("--out-dir", type=Path, default=PLATEAU_DIR, help="output dir (default: plateau/)")
    ap.add_argument("--skip-train", action="store_true", help="skip training, only plot from existing results")
    ap.add_argument("--plot-logratio-only", action="store_true", help="only plot log-ratio GIFs and trajectories for run dirs that have logratio_values.npy; then exit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_logratio_only:
        for run_dir in sorted(out_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            if not (run_dir / "logratio_values.npy").exists():
                continue
            try:
                plot_logratio_trajectories(run_dir, run_name, plot_filename="logratio_trajectories.png")
            except Exception as e:
                print(f"skip plot {run_name}: {e}")
            if (run_dir / "logratio_values_2nd.npy").exists():
                try:
                    plot_logratio_trajectories(
                        run_dir,
                        run_name,
                        layer_suffix=" (2nd layer)",
                        values_filename="logratio_values_2nd.npy",
                        plot_filename="logratio_trajectories_2nd.png",
                    )
                except Exception as e:
                    print(f"skip plot 2nd {run_name}: {e}")
        print("done (log-ratio plots only). output:", out_dir)
        return

    if args.config.exists():
        with open(args.config) as f:
            base_config = json.load(f)
    else:
        base_config = {
            "factor": 3,
            "n_train": 768,
            "num_layers": 3,
            "hidden_width": SWEEP_WIDTH,
            "hidden_rank": SWEEP_RANK,
        }

    if not args.skip_train:
        results_grid: list[dict] = []
        for lr in LR_LIST:
            for bs in BATCH_SIZES:
                if bs > base_config.get("n_train", 768):
                    continue
                run_name = f"lr{lr:.0e}_bs{bs}".replace(".", "")
                run_dir = out_dir / run_name
                max_epochs_run = min(args.max_epochs, max(1, int(MAX_EPOCHS_PER_LR_FACTOR / lr)))
                print(f"run {run_name} (max_epochs={max_epochs_run})")
                res = run_one(
                    base_config=base_config,
                    lr=lr,
                    batch_size=bs,
                    output_dir=run_dir,
                    threshold=args.threshold,
                    max_epochs=max_epochs_run,
                    seed=SEED,
                )
                res["lr"] = lr
                res["batch_size"] = bs
                res["run_name"] = run_name
                results_grid.append({k: v for k, v in res.items() if k != "all_losses"})
        with open(out_dir / "all_results.json", "w") as f:
            json.dump(results_grid, f, indent=2)
    else:
        results_grid = []
        for lr in LR_LIST:
            for bs in BATCH_SIZES:
                run_name = f"lr{lr:.0e}_bs{bs}".replace(".", "")
                run_dir = out_dir / run_name
                res_path = run_dir / "results.json"
                if not res_path.exists():
                    continue
                with open(res_path) as f:
                    res = json.load(f)
                res["lr"] = lr
                res["batch_size"] = bs
                res["run_name"] = run_name
                results_grid.append(res)
        if not results_grid:
            print("no results found; run without --skip-train first")
            return

    # we make log-ratio GIF and trajectory plot for each run (last and 2nd low-rank layer when present)
    for lr in LR_LIST:
        for bs in BATCH_SIZES:
            if bs > base_config.get("n_train", 768):
                continue
            run_name = f"lr{lr:.0e}_bs{bs}".replace(".", "")
            run_dir = out_dir / run_name
            if (run_dir / "logratio_values.npy").exists():
                try:
                    plot_logratio_trajectories(run_dir, run_name, plot_filename="logratio_trajectories.png")
                except Exception as e:
                    print(f"skip plot {run_name}: {e}")
            if (run_dir / "logratio_values_2nd.npy").exists():
                try:
                    plot_logratio_trajectories(
                        run_dir,
                        run_name,
                        layer_suffix=" (2nd layer)",
                        values_filename="logratio_values_2nd.npy",
                        plot_filename="logratio_trajectories_2nd.png",
                    )
                except Exception as e:
                    print(f"skip plot 2nd {run_name}: {e}")

    # plot: epochs_to_escape vs lr (log-log), one curve per batch size
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
    mpl.rcParams["axes.formatter.use_mathtext"] = False
    mpl.rcParams["font.family"] = "serif"
    try:
        fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        bs128_points = None
        for bs in BATCH_SIZES:
            points = [(r["lr"], r["epochs_to_escape"]) for r in results_grid if r["batch_size"] == bs and r.get("escaped") and r.get("epochs_to_escape") is not None]
            if not points:
                continue
            lrs = [p[0] for p in points]
            ep = [p[1] for p in points]
            ax1.loglog(lrs, ep, "o-", label=f"bs={bs}", linewidth=1.5, markersize=6)
            if bs == 128:
                bs128_points = (lrs, ep)
        # scaling law for bs=128: fit epochs ~ lr^exponent in log-log (log(ep) = a + exponent*log(lr))
        if bs128_points is not None and len(bs128_points[0]) >= 2:
            lrs_b = np.array(bs128_points[0], dtype=float)
            ep_b = np.array(bs128_points[1], dtype=float)
            log_lr = np.log(lrs_b)
            log_ep = np.log(ep_b)
            slope, intercept = np.polyfit(log_lr, log_ep, 1)
            lr_fit = np.linspace(min(lrs_b), max(lrs_b), 50)
            ep_fit = np.exp(intercept) * (lr_fit ** slope)
            ax1.loglog(lr_fit, ep_fit, "--", color="black", linewidth=1.2, alpha=0.7)
            ax1.text(0.05, 0.95, f"bs=128 scaling: epochs ~ lr^{slope:.2f}", transform=ax1.transAxes, fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax1.set_xlabel("learning rate")
        ax1.set_ylabel("epochs to escape plateau")
        ax1.set_title(f"Time to escape plateau vs LR (loss < {args.threshold:.2e})")
        ax1.legend()
        ax1.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        fig1.savefig(out_dir / "epochs_to_escape_vs_lr.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print("saved", out_dir / "epochs_to_escape_vs_lr.png")
    except Exception as e:
        print(f"skip epochs_to_escape_vs_lr.png: {e}")

    # plot: epochs_to_escape vs batch size, one curve per lr; scaling law for lr=1e-3 and bs>10
    try:
        fig_bs, ax_bs = plt.subplots(1, 1, figsize=(7, 5))
        lr1e3_points = None
        for lr in LR_LIST:
            points = [(r["batch_size"], r["epochs_to_escape"]) for r in results_grid if r["lr"] == lr and r.get("escaped") and r.get("epochs_to_escape") is not None]
            if not points:
                continue
            bss = [p[0] for p in points]
            ep = [p[1] for p in points]
            ax_bs.loglog(bss, ep, "o-", label=f"lr={lr:.0e}", linewidth=1.5, markersize=6)
            if lr == 1e-3:
                lr1e3_points = (bss, ep)
        # scaling law for lr=1e-3, bs>10: fit epochs ~ bs^exponent in log-log
        if lr1e3_points is not None:
            bss_b = np.array([b for b in lr1e3_points[0] if b > 10], dtype=float)
            ep_b = np.array([ep for b, ep in zip(lr1e3_points[0], lr1e3_points[1]) if b > 10], dtype=float)
            if len(bss_b) >= 2:
                log_bs = np.log(bss_b)
                log_ep = np.log(ep_b)
                slope, intercept = np.polyfit(log_bs, log_ep, 1)
                bs_fit = np.linspace(min(bss_b), max(bss_b), 50)
                ep_fit = np.exp(intercept) * (bs_fit ** slope)
                ax_bs.loglog(bs_fit, ep_fit, "--", color="black", linewidth=1.2, alpha=0.7)
                ax_bs.text(0.05, 0.95, f"lr=1e-3 (bs>10): epochs ~ bs^{slope:.2f}", transform=ax_bs.transAxes, fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax_bs.set_xlabel("batch size")
        ax_bs.set_ylabel("epochs to escape plateau")
        ax_bs.set_title(f"Time to escape plateau vs batch size (loss < {args.threshold:.2e})")
        ax_bs.legend()
        ax_bs.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        fig_bs.savefig(out_dir / "epochs_to_escape_vs_bs.png", dpi=150, bbox_inches="tight")
        plt.close(fig_bs)
        print("saved", out_dir / "epochs_to_escape_vs_bs.png")
    except Exception as e:
        print(f"skip epochs_to_escape_vs_bs.png: {e}")

    # histogram of norm_diff (only escaped runs)
    norms = [r["norm_diff"] for r in results_grid if r.get("escaped") and isinstance(r.get("norm_diff"), (int, float)) and not math.isnan(r["norm_diff"])]
    if norms:
        try:
            fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
            ax2.hist(norms, bins=min(20, max(5, len(norms))), color="steelblue", alpha=0.8, edgecolor="white")
            ax2.set_xlabel("||theta_escape - theta_init||_2")
            ax2.set_ylabel("count")
            ax2.set_title("Histogram of param norm at plateau escape")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            fig2.savefig(out_dir / "histogram_norm_diff.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print("saved", out_dir / "histogram_norm_diff.png")
        except Exception as e:
            print(f"skip histogram_norm_diff.png: {e}")
    else:
        print("no escaped runs with norm_diff for histogram")

    print("done. output:", out_dir)


if __name__ == "__main__":
    main()

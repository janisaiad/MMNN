#!/usr/bin/env python3
import argparse
import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # we add repo root to import path

from experiments.table.mmnn_vs import MMNN  # we reuse the exact MMNN used by training scripts

def configure_tex_style() -> None:
    plt.rcParams["figure.figsize"] = [6, 6]
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "normal"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["font.size"] = 22
    mpl.rcParams["axes.formatter.limits"] = (-6, 6)
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["font.family"] = "STIXGeneral"
    mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
    mpl.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
    mpl.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.top"] = True

configure_tex_style()


@dataclass(frozen=True)
class RunConfig:
    run_dir: Path
    num_layers: int
    hidden_width: int
    hidden_rank: int
    input_rank: int
    output_rank: int
    use_resnet: bool
    interval: Tuple[float, float]
    dtype: torch.dtype


def _torch_dtype_from_cfg(dtype_str: str) -> torch.dtype:
    dtype_name = dtype_str.split(".")[-1].strip()  # we parse "torch.float32" to "float32"
    if not hasattr(torch, dtype_name):  # we validate dtype exists
        raise ValueError(f"unsupported dtype string: {dtype_str}")
    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):  # we validate parsed object is a dtype
        raise ValueError(f"invalid dtype resolved from: {dtype_str}")
    return dtype


def load_run_config(run_dir: Path) -> RunConfig:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():  # we ensure config exists
        raise FileNotFoundError(f"missing config.json in {run_dir}")
    with cfg_path.open("r") as f:
        cfg = json.load(f)
    interval = cfg.get("interval", [-1.0, 1.0])
    if not (isinstance(interval, list) and len(interval) == 2):  # we validate interval shape
        raise ValueError(f"invalid interval in {cfg_path}: {interval}")
    dtype = _torch_dtype_from_cfg(str(cfg.get("dtype", "torch.float32")))
    return RunConfig(
        run_dir=run_dir,
        num_layers=int(cfg["num_layers"]),
        hidden_width=int(cfg["hidden_width"]),
        hidden_rank=int(cfg["hidden_rank"]),
        input_rank=int(cfg.get("input_rank", 1)),
        output_rank=int(cfg.get("output_rank", 1)),
        use_resnet=bool(cfg.get("use_resnet", False)),
        interval=(float(interval[0]), float(interval[1])),
        dtype=dtype,
    )


def load_state_dict(run_dir: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    model_path = run_dir / "model_parameters.pth"
    checkpoint_path = run_dir / "checkpoint.pth"
    if model_path.exists():  # we prefer the direct state dict
        state = torch.load(model_path, map_location=device)
    elif checkpoint_path.exists():  # we fall back to checkpoint format
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
    else:
        raise FileNotFoundError(f"missing model_parameters.pth/checkpoint.pth in {run_dir}")
    if not isinstance(state, dict):  # we validate load result
        raise ValueError(f"unexpected state type: {type(state)} in {run_dir}")
    return state


def build_model(cfg: RunConfig, device: torch.device) -> MMNN:
    ranks = [cfg.input_rank] + [cfg.hidden_rank] * cfg.num_layers + [cfg.output_rank]
    widths = [cfg.hidden_width] * (cfg.num_layers + 1)  # we match training scripts: hidden blocks + final output block
    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=str(device),
        ResNet=cfg.use_resnet,
        fixWb=False,
    )
    model.to(device)
    model.eval()
    return model


def compute_module_outputs(model: MMNN, x: torch.Tensor) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    current = x
    with torch.no_grad():
        for i, fc in enumerate(model.fcs):
            current = fc(current)
            if i % 2 == 0:
                current = torch.relu(current)
            outputs.append(current)
    return outputs


def count_local_minima_per_component(y2d: np.ndarray, value_threshold: float) -> np.ndarray:
    if y2d.ndim == 1:  # we normalize to 2d
        y2d = y2d.reshape(-1, 1)
    if y2d.ndim != 2:  # we validate 2d
        raise ValueError("layer output must be 2d after reshape")
    if y2d.shape[0] < 3:  # we need at least 3 points to form a strict local minimum
        return np.zeros((y2d.shape[1],), dtype=int)
    y0 = y2d[:-2, :]
    y1 = y2d[1:-1, :]
    y2 = y2d[2:, :]
    is_min = (y1 < y0) & (y1 < y2)
    is_pos = y1 > float(value_threshold)
    return np.sum(is_min & is_pos, axis=0).astype(int)


def iter_run_dirs(runs_root: Path) -> Iterable[Path]:
    if not runs_root.exists():  # we validate root exists
        raise FileNotFoundError(f"runs_root does not exist: {runs_root}")
    for p in sorted(runs_root.iterdir()):
        if p.is_dir() and (p / "config.json").exists() and ((p / "model_parameters.pth").exists() or (p / "checkpoint.pth").exists()):
            yield p


def summarize_counts(counts_by_run: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    if len(counts_by_run) == 0:  # we handle empty input
        return np.array([]), np.array([])
    arr = np.array(counts_by_run, dtype=float)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def plot_mean_with_std(mean: np.ndarray, std: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    x = np.arange(1, len(mean) + 1)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(x, mean, "b-", linewidth=2, label="mean")
    ax.fill_between(x, mean - std, mean + std, color="b", alpha=0.2, label="±1 std")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_mean_curve(mean: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str, label: str = "mean") -> None:
    x = np.arange(1, len(mean) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, mean, "b-", linewidth=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(counts_by_run: List[List[int]], out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    if len(counts_by_run) == 0:  # we skip on empty input
        return
    arr = np.array(counts_by_run, dtype=float)
    plt.figure(figsize=(10, 5))
    plt.boxplot([arr[:, i] for i in range(arr.shape[1])], showfliers=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_histogram_across_layers(layer_counts: List[int], out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    if len(layer_counts) == 0:  # we skip on empty input
        return
    x = np.array(layer_counts, dtype=float)
    maxv = int(np.max(x)) if x.size > 0 else 0
    bins = np.arange(-0.5, maxv + 1.5, 1.0)
    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=bins, color="C0", alpha=0.8, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_boxplot_from_arrays(per_layer_arrays: List[np.ndarray], out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    if len(per_layer_arrays) == 0:  # we skip on empty input
        return
    data = [np.asarray(a, dtype=float) for a in per_layer_arrays]
    plt.figure(figsize=(10, 5))
    plt.boxplot(data, showfliers=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_histogram_from_values(values: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    x = np.asarray(values, dtype=float)
    if x.size == 0:  # we skip on empty input
        return
    maxv = int(np.max(x))
    bins = np.arange(-0.5, maxv + 1.5, 1.0)
    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=bins, color="C0", alpha=0.8, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_histogram_linear(values: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str, bins: int) -> None:
    x = np.asarray(values, dtype=float)
    if x.size == 0:  # we skip on empty input
        return
    b = int(max(5, int(bins)))
    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=b, color="C0", alpha=0.8, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_boxplot_from_arrays_limited(per_layer_arrays: List[np.ndarray], out_path: Path, title: str, xlabel: str, ylabel: str, max_layers: int) -> None:
    if len(per_layer_arrays) == 0:  # we skip on empty input
        return
    n = int(min(int(max_layers), int(len(per_layer_arrays))))
    data = [np.asarray(per_layer_arrays[i], dtype=float) for i in range(n)]
    plt.figure(figsize=(max(10, int(0.6 * n)), 5))
    plt.boxplot(data, showfliers=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_signal_with_minima(x: np.ndarray, y: np.ndarray, minima_idx: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(x, y, "b-", linewidth=1.5, label="signal")
    if minima_idx.size > 0:
        plt.plot(x[minima_idx], y[minima_idx], "ro", markersize=3, label="local minima")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fft_power(freq: np.ndarray, power: np.ndarray, out_path: Path, title: str) -> None:
    eps = 1e-30
    plt.figure(figsize=(9, 4))
    plt.plot(freq, np.log10(power + eps), "k-", linewidth=1.0)
    plt.xlabel("frequency (cycles per x-unit)")
    plt.ylabel("log10 power")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_strict_local_minima_indices(y: np.ndarray, value_threshold: float) -> np.ndarray:
    if y.ndim != 1:  # we validate 1d input
        raise ValueError("y must be 1d")
    if y.size < 3:  # we return empty if too short
        return np.array([], dtype=int)
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    is_min = (y1 < y0) & (y1 < y2)
    is_pos = y1 > float(value_threshold)
    idx = np.where(is_min & is_pos)[0] + 1
    return idx.astype(int)


def compute_fft_metrics(y: np.ndarray, dx: float, window: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if y.ndim != 1:  # we validate 1d input
        raise ValueError("y must be 1d")
    if y.size < 4:  # we keep minimal length for fft
        freq = np.array([], dtype=float)
        power = np.array([], dtype=float)
        return freq, power, {"dominant_freq": 0.0, "spectral_centroid": 0.0, "spectral_rolloff_95": 0.0}
    y0 = y.astype(np.float64) - float(np.mean(y))  # we remove mean
    if window == "hann":
        w = np.hanning(y0.size)
        y0 = y0 * w
    elif window != "none":
        raise ValueError(f"unsupported window: {window}")
    spec = np.fft.rfft(y0)
    power = (np.abs(spec) ** 2).astype(np.float64)
    freq = np.fft.rfftfreq(y0.size, d=float(dx)).astype(np.float64)
    if power.size <= 1:
        return freq, power, {"dominant_freq": 0.0, "spectral_centroid": 0.0, "spectral_rolloff_95": 0.0}
    p = power.copy()
    p[0] = 0.0  # we ignore dc
    total = float(np.sum(p))
    if total <= 0.0:
        return freq, power, {"dominant_freq": 0.0, "spectral_centroid": 0.0, "spectral_rolloff_95": 0.0}
    dominant_freq = float(freq[int(np.argmax(p))])
    spectral_centroid = float(np.sum(freq * p) / total)
    cdf = np.cumsum(p) / total
    roll_idx = int(np.searchsorted(cdf, 0.95))
    spectral_rolloff_95 = float(freq[min(roll_idx, freq.size - 1)])
    return freq, power, {"dominant_freq": dominant_freq, "spectral_centroid": spectral_centroid, "spectral_rolloff_95": spectral_rolloff_95}


def get_layer_component_signal(module_outputs: List[torch.Tensor], target: str, layer_index_1based: int, component_index_1based: int) -> np.ndarray:
    if layer_index_1based <= 0:  # we validate indices
        raise ValueError("layer_index must be >= 1")
    if component_index_1based <= 0:  # we validate indices
        raise ValueError("component_index must be >= 1")
    block_idx = layer_index_1based - 1
    module_idx = (2 * block_idx) if target == "relu_width" else (2 * block_idx + 1)
    if module_idx < 0 or module_idx >= len(module_outputs):  # we validate module index
        raise ValueError(f"layer_index out of range for this model: layer_index={layer_index_1based}")
    y2d = module_outputs[module_idx].detach().cpu().numpy()
    if y2d.ndim == 1:
        y2d = y2d.reshape(-1, 1)
    comp_idx = component_index_1based - 1
    if comp_idx < 0 or comp_idx >= y2d.shape[1]:  # we validate component index
        raise ValueError(f"component_index out of range: component_index={component_index_1based}, available={y2d.shape[1]}")
    return y2d[:, comp_idx].astype(np.float64)

def get_leap_layer_output(module_outputs: List[torch.Tensor], leap_layer_idx_1based: int) -> np.ndarray:
    if leap_layer_idx_1based <= 0:  # we validate indices
        raise ValueError("leap_layer_idx must be >= 1")
    idx = leap_layer_idx_1based - 1
    if idx < 0 or idx >= len(module_outputs):  # we validate range
        raise ValueError(f"leap_layer_idx out of range for this model: {leap_layer_idx_1based}")
    y2d = module_outputs[idx].detach().cpu().numpy()
    if y2d.ndim == 1:
        y2d = y2d.reshape(-1, 1)
    return y2d.astype(np.float64)

def plot_leap_layer_components_grid(x: np.ndarray, y2d: np.ndarray, out_path: Path, title: str, max_components: int) -> None:
    if y2d.ndim != 2:  # we validate 2d
        raise ValueError("y2d must be 2d")
    k = int(min(int(max_components), int(y2d.shape[1])))
    if k <= 0:  # we skip if empty
        return
    n_rows = int(np.ceil(np.sqrt(k)))
    n_cols = int(np.ceil(k / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = np.asarray(axes).reshape(n_rows, n_cols)
    fig.suptitle(title, fontsize=16)
    for idx in range(n_rows * n_cols):
        i = idx // n_cols
        j = idx % n_cols
        ax = axes[i, j]
        if idx < k:
            ax.plot(x, y2d[:, idx], "b-", linewidth=1)
            ax.set_title(f"Component {idx + 1}")
            ax.grid(True, alpha=0.3)
            ax.set_xticks([float(x[0]), 0.0, float(x[-1])])
        else:
            ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="we count strictly positive local minima of layer-wise partial functions from saved MMNN weights")
    parser.add_argument("--runs_root", type=str, default=None, help="we set the directory containing run folders (default: sibling ../mmnn_training)")
    parser.add_argument("--run_dir", type=str, default=None, help="we analyze a single run directory instead of scanning runs_root")
    parser.add_argument("--out_dir", type=str, default=None, help="we set output directory (default: experiments/partialfunctionanalysis/output)")
    parser.add_argument("--grid_size", type=int, default=1000, help="we set number of x grid points for minima counting")
    parser.add_argument("--value_threshold", type=float, default=1e-4, help="we only count minima where function value is > value_threshold")
    parser.add_argument("--target", type=str, default="relu_width", choices=["relu_width", "rank"], help="we choose which intermediate representation to analyze")
    parser.add_argument("--max_hidden_rank", type=int, default=50, help="we only include runs with hidden_rank < max_hidden_rank (set <=0 to disable)")
    parser.add_argument("--max_hidden_width", type=int, default=0, help="we only include runs with hidden_width < max_hidden_width (set <=0 to disable)")
    parser.add_argument("--only_num_layers", type=str, default="", help="we optionally restrict to num_layers in this comma list, e.g. \"6,8,12\"")
    parser.add_argument("--only_hidden_width", type=str, default="", help="we optionally restrict to hidden_width in this comma list, e.g. \"128,256\"")
    parser.add_argument("--only_hidden_rank", type=str, default="", help="we optionally restrict to hidden_rank in this comma list, e.g. \"20\"")
    parser.add_argument("--include_output_block", action="store_true", help="we include the final output block (default: excluded)")
    parser.add_argument("--single_layer", type=int, default=0, help="we optionally analyze a single layer index (1-based) for one component and export fft plots")
    parser.add_argument("--single_component", type=int, default=0, help="we optionally analyze a single component/neuron index (1-based) for single_layer")
    parser.add_argument("--single_pick", type=str, default="none", choices=["none", "max"], help="we optionally pick a component automatically for single_layer (max picks the neuron with max minima)")
    parser.add_argument("--fft_window", type=str, default="hann", choices=["hann", "none"], help="we optionally apply a window before fft to reduce leakage")
    parser.add_argument("--compute_fft", action="store_true", help="we compute and export fft plots/metrics (default: off)")
    parser.add_argument("--leap_layer_idx", type=int, default=0, help="we optionally extract a leap.py-style layer output (1-based fcs index)")
    parser.add_argument("--leap_component", type=int, default=0, help="we optionally extract one component from leap_layer_idx (1-based)")
    parser.add_argument("--leap_components_first", type=int, default=0, help="we optionally export the first K components (and minima histogram) for leap_layer_idx")
    parser.add_argument("--export_leap_grid", action="store_true", help="we export a leap-style grid plot for leap_layer_idx")
    parser.add_argument("--export_leap_max_components", type=int, default=36, help="we cap number of components in leap grid plots")
    parser.add_argument("--leap_all_layers", action="store_true", help="we compute leap-style minima distributions for all layers (1..#fcs)")
    parser.add_argument("--hist_bins", type=int, default=50, help="we set number of bins for histograms when counts are large")
    parser.add_argument("--export_leap_grids_all", action="store_true", help="we also export leap-style component grids for all layers (can be heavy)")
    parser.add_argument("--boxplot_max_layers", type=int, default=60, help="we cap layers shown in aggregated boxplot for readability")
    parser.add_argument("--group_by", type=str, default="L,R", help="we group runs for comparative study; comma list from {L,W,R}, e.g. \"L,R\" or \"L,R,W\"")
    parser.add_argument("--group_hist_mode", type=str, default="both", choices=["pooled", "per_layer", "both"], help="we choose which grouped histograms to export")
    parser.add_argument("--device", type=str, default="cpu", help="we choose device for forward passes (cpu recommended)")
    args = parser.parse_args()

    def _parse_int_set(s: str) -> Optional[set[int]]:
        s = str(s).strip()
        if s == "":  # we treat empty string as no filter
            return None
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        vals: set[int] = set()
        for p in parts:
            vals.add(int(p))
        return vals

    repo_root = Path(__file__).resolve().parents[2]
    default_runs_root = repo_root.parent / "mmnn_training"
    runs_root = Path(args.runs_root).expanduser() if args.runs_root is not None else default_runs_root
    out_dir = Path(args.out_dir).expanduser() if args.out_dir is not None else (repo_root / "experiments" / "partialfunctionanalysis" / "output")
    device = torch.device(args.device)

    run_dirs: List[Path]
    if args.run_dir is not None:
        run_dirs = [Path(args.run_dir).expanduser()]
    else:
        run_dirs = list(iter_run_dirs(runs_root))

    selected_runs: List[RunConfig] = []
    only_L = _parse_int_set(args.only_num_layers)
    only_W = _parse_int_set(args.only_hidden_width)
    only_R = _parse_int_set(args.only_hidden_rank)
    for d in run_dirs:
        try:
            cfg = load_run_config(d)
        except Exception:
            continue
        if args.max_hidden_rank > 0 and not (cfg.hidden_rank < args.max_hidden_rank):  # we enforce strict bound if requested
            continue
        if args.max_hidden_width > 0 and not (cfg.hidden_width < args.max_hidden_width):  # we enforce strict bound if requested
            continue
        if only_L is not None and int(cfg.num_layers) not in only_L:  # we filter by depth if requested
            continue
        if only_W is not None and int(cfg.hidden_width) not in only_W:  # we filter by width if requested
            continue
        if only_R is not None and int(cfg.hidden_rank) not in only_R:  # we filter by rank if requested
            continue
        selected_runs.append(cfg)

    if len(selected_runs) == 0:
        raise RuntimeError("no runs matched your filters; try lowering restrictions or passing --run_dir")

    counts_rank_by_run: List[List[int]] = []
    counts_output_by_run: List[int] = []
    per_run_records: List[Dict[str, object]] = []

    for cfg in selected_runs:
        state = load_state_dict(cfg.run_dir, device)
        model = build_model(cfg, device)
        model.load_state_dict(state, strict=True)

        x_np = np.linspace(cfg.interval[0], cfg.interval[1], int(args.grid_size), dtype=np.float64)
        dx = float(x_np[1] - x_np[0]) if x_np.size >= 2 else 1.0
        x_t = torch.tensor(x_np.reshape(-1, 1), dtype=cfg.dtype, device=device)
        module_outputs = compute_module_outputs(model, x_t)

        depth_blocks = len(model.widths)
        blocks_to_analyze = depth_blocks if args.include_output_block else min(int(cfg.num_layers), depth_blocks)
        per_layer_means: List[float] = []
        per_layer_stds: List[float] = []
        per_layer_per_component: List[np.ndarray] = []
        for block_idx in range(blocks_to_analyze):
            module_idx = (2 * block_idx) if args.target == "relu_width" else (2 * block_idx + 1)
            if module_idx >= len(module_outputs):  # we guard against unexpected shapes
                break
            y = module_outputs[module_idx].detach().cpu().numpy()
            per_component = count_local_minima_per_component(y, float(args.value_threshold))
            per_layer_per_component.append(per_component)
            per_layer_means.append(float(np.mean(per_component)) if per_component.size > 0 else 0.0)
            per_layer_stds.append(float(np.std(per_component)) if per_component.size > 0 else 0.0)

        y_out = module_outputs[-1].detach().cpu().numpy()
        out_total = int(np.sum(count_local_minima_per_component(y_out, float(args.value_threshold))))

        counts_rank_by_run.append([float(v) for v in per_layer_means])
        counts_output_by_run.append(out_total)

        pooled_components = np.concatenate(per_layer_per_component, axis=0) if len(per_layer_per_component) > 0 else np.array([], dtype=int)
        single_selected: Dict[str, int] = {}
        single_metrics: Dict[str, float] = {}
        if int(args.single_layer) > 0:
            if int(args.single_component) <= 0 and str(args.single_pick) == "none":  # we require a selection rule
                raise ValueError("single_layer was set but neither single_component nor single_pick was provided")
            layer_idx = int(args.single_layer)
            if layer_idx < 1 or layer_idx > len(per_layer_per_component):  # we validate layer range
                raise ValueError(f"single_layer out of range for this run: {layer_idx}")
            if str(args.single_pick) == "max":
                per_comp = per_layer_per_component[layer_idx - 1]
                comp_idx = (int(np.argmax(per_comp)) + 1) if per_comp.size > 0 else 1
            else:
                comp_idx = int(args.single_component)
            y_single = get_layer_component_signal(module_outputs, str(args.target), layer_idx, comp_idx)
            minima_idx = compute_strict_local_minima_indices(y_single, float(args.value_threshold))
            single_selected = {"layer_index": int(layer_idx), "component_index": int(comp_idx)}
            single_metrics = {"single_minima_count": float(minima_idx.size)}
            run_subdir = cfg.run_dir.name.replace("/", "_")
            single_dir = out_dir / "single_component" / run_subdir  # we separate runs to avoid collisions
            plot_signal_with_minima(
                x_np,
                y_single,
                minima_idx,
                single_dir / f"single_component_layer{layer_idx}_comp{comp_idx}_signal.png",
                title=f"single component signal (target={args.target}, layer={layer_idx}, comp={comp_idx})",
            )
            if bool(args.compute_fft):
                freq, power, fftm = compute_fft_metrics(y_single, dx=dx, window=str(args.fft_window))
                single_metrics = {
                    **single_metrics,
                    "dominant_freq": float(fftm["dominant_freq"]),
                    "spectral_centroid": float(fftm["spectral_centroid"]),
                    "spectral_rolloff_95": float(fftm["spectral_rolloff_95"]),
                }
                if freq.size > 0:
                    plot_fft_power(
                        freq,
                        power,
                        single_dir / f"single_component_layer{layer_idx}_comp{comp_idx}_fft.png",
                        title=f"fft power (window={args.fft_window})",
                    )
        leap_selected: Dict[str, int] = {}
        leap_metrics: Dict[str, float] = {}
        if int(args.leap_layer_idx) > 0:
            leap_layer_idx = int(args.leap_layer_idx)
            y_leap = get_leap_layer_output(module_outputs, leap_layer_idx)
            run_subdir = cfg.run_dir.name.replace("/", "_")
            leap_dir = out_dir / "leap_extraction" / run_subdir
            if bool(args.export_leap_grid):
                plot_leap_layer_components_grid(
                    x_np,
                    y_leap,
                    leap_dir / f"leap_layer{leap_layer_idx}_components.png",
                    title=f"layer {leap_layer_idx} components (max {int(args.export_leap_max_components)})",
                    max_components=int(args.export_leap_max_components),
                )
            if int(args.leap_components_first) > 0:
                k = int(args.leap_components_first)
                k = int(min(k, int(y_leap.shape[1])))
                if k <= 0:
                    raise ValueError("leap_components_first resolved to <= 0")
                minima_counts: List[int] = []
                for comp_idx in range(1, k + 1):
                    y_single = y_leap[:, comp_idx - 1]
                    minima_idx = compute_strict_local_minima_indices(y_single, float(args.value_threshold))
                    minima_counts.append(int(minima_idx.size))
                    plot_signal_with_minima(
                        x_np,
                        y_single,
                        minima_idx,
                        leap_dir / f"leap_layer{leap_layer_idx}_comp{comp_idx}_signal.png",
                        title=f"single component (layer={leap_layer_idx}, comp={comp_idx})",
                    )
                plot_histogram_linear(
                    np.asarray(minima_counts, dtype=int),
                    leap_dir / f"leap_layer{leap_layer_idx}_hist_minima_first{k}_components.png",
                    title=f"histogram of minima counts (layer {leap_layer_idx}, first {k} components)",
                    xlabel="# strictly-positive local minima per component",
                    ylabel="count of components",
                    bins=min(int(args.hist_bins), max(5, k)),
                )
                leap_selected = {"leap_layer_idx": int(leap_layer_idx), "first_k_components": int(k)}
                leap_metrics = {"mean_minima_first_k": float(np.mean(np.asarray(minima_counts, dtype=float)))}
            if int(args.leap_component) > 0:
                comp_idx = int(args.leap_component)
                if comp_idx < 1 or comp_idx > int(y_leap.shape[1]):  # we validate component index
                    raise ValueError(f"leap_component out of range: {comp_idx}, available={y_leap.shape[1]}")
                y_single = y_leap[:, comp_idx - 1]
                minima_idx = compute_strict_local_minima_indices(y_single, float(args.value_threshold))
                leap_selected = {"leap_layer_idx": int(leap_layer_idx), "component_index": int(comp_idx)}
                leap_metrics = {"single_minima_count": float(minima_idx.size)}
                plot_signal_with_minima(
                    x_np,
                    y_single,
                    minima_idx,
                    leap_dir / f"leap_layer{leap_layer_idx}_comp{comp_idx}_signal.png",
                    title=f"single component (layer={leap_layer_idx}, comp={comp_idx})",
                )
                if bool(args.compute_fft):
                    freq, power, fftm = compute_fft_metrics(y_single, dx=dx, window=str(args.fft_window))
                    leap_metrics = {
                        **leap_metrics,
                        "dominant_freq": float(fftm["dominant_freq"]),
                        "spectral_centroid": float(fftm["spectral_centroid"]),
                        "spectral_rolloff_95": float(fftm["spectral_rolloff_95"]),
                    }
                    if freq.size > 0:
                        plot_fft_power(
                            freq,
                            power,
                            leap_dir / f"leap_layer{leap_layer_idx}_comp{comp_idx}_fft.png",
                            title=f"fft power (window={args.fft_window})",
                        )
        leap_all_records: Dict[str, object] = {}
        if bool(args.leap_all_layers):
            run_subdir = cfg.run_dir.name.replace("/", "_")
            leap_root = out_dir / "leap_all_layers" / run_subdir
            all_layer_arrays: List[np.ndarray] = []
            all_layer_means: List[float] = []
            pooled_all: List[int] = []
            for leap_layer_idx in range(1, len(module_outputs) + 1):
                y_leap = get_leap_layer_output(module_outputs, leap_layer_idx)
                per_comp = count_local_minima_per_component(y_leap, float(args.value_threshold))
                all_layer_arrays.append(per_comp)
                all_layer_means.append(float(np.mean(per_comp)) if per_comp.size > 0 else 0.0)
                pooled_all.extend([int(v) for v in per_comp.tolist()])
                layer_dir = leap_root / f"layer_{leap_layer_idx:02d}"
                plot_histogram_linear(
                    per_comp,
                    layer_dir / f"hist_minima_per_component_layer{leap_layer_idx:02d}.png",
                    title=f"minima per component histogram (layer {leap_layer_idx})",
                    xlabel="# strictly-positive local minima per component",
                    ylabel="count of components",
                    bins=int(args.hist_bins),
                )
                if bool(args.export_leap_grids_all):
                    plot_leap_layer_components_grid(
                        x_np,
                        y_leap,
                        layer_dir / f"leap_layer{leap_layer_idx:02d}_components.png",
                        title=f"layer {leap_layer_idx} components (max {int(args.export_leap_max_components)})",
                        max_components=int(args.export_leap_max_components),
                    )
            pooled_all_arr = np.asarray(pooled_all, dtype=int)
            plot_mean_curve(
                np.asarray(all_layer_means, dtype=float),
                leap_root / "mean_minima_per_component_vs_layer.png",
                title=f"L={cfg.num_layers}, W={cfg.hidden_width}, R={cfg.hidden_rank}",
                xlabel="layer index",
                ylabel="mean minima per component",
            )
            plot_boxplot_from_arrays_limited(
                all_layer_arrays,
                leap_root / "boxplot_minima_per_component_vs_layer.png",
                title="distribution over components vs layer",
                xlabel="layer index",
                ylabel="# strictly-positive local minima per component",
                max_layers=int(args.boxplot_max_layers),
            )
            plot_histogram_linear(
                pooled_all_arr,
                leap_root / "hist_minima_pooled_all_layers.png",
                title="pooled minima histogram across all layers/components",
                xlabel="# strictly-positive local minima per component",
                ylabel="count of components (pooled)",
                bins=int(args.hist_bins),
            )
            leap_all_records = {
                "n_leap_layers": int(len(module_outputs)),
                "mean_minima_per_component_by_leap_layer": [float(v) for v in all_layer_means],
                "minima_per_component_by_leap_layer": [a.tolist() for a in all_layer_arrays],
                "pooled_minima_per_component": pooled_all_arr.tolist(),
            }
        per_run_records.append(
            {
                "run_dir": str(cfg.run_dir),
                "num_layers": cfg.num_layers,
                "hidden_width": cfg.hidden_width,
                "hidden_rank": cfg.hidden_rank,
                "depth_blocks": depth_blocks,
                "grid_size": int(args.grid_size),
                "value_threshold": float(args.value_threshold),
                "include_output_block": bool(args.include_output_block),
                "target": str(args.target),
                "layer_local_minima_per_component": [pc.tolist() for pc in per_layer_per_component],
                "layer_local_minima_mean_per_component": per_layer_means,
                "layer_local_minima_std_per_component": per_layer_stds,
                "pooled_local_minima_per_component": pooled_components.tolist(),
                "single_component_selected": single_selected,
                "single_component_metrics": single_metrics,
                "leap_component_selected": leap_selected,
                "leap_component_metrics": leap_metrics,
                "leap_all_layers": leap_all_records,
                "final_output_positive_local_minima_total": int(out_total),
            }
        )

    min_len = min(len(v) for v in counts_rank_by_run)
    counts_rank_by_run = [v[:min_len] for v in counts_rank_by_run]

    mean_rank, std_rank = summarize_counts(counts_rank_by_run)

    summary = {
        "n_runs": len(selected_runs),
        "filters": {
            "runs_root": str(runs_root),
            "max_hidden_rank": int(args.max_hidden_rank),
            "max_hidden_width": int(args.max_hidden_width),
            "only_num_layers": str(args.only_num_layers),
            "only_hidden_width": str(args.only_hidden_width),
            "only_hidden_rank": str(args.only_hidden_rank),
        },
        "grid_size": int(args.grid_size),
        "value_threshold": float(args.value_threshold),
        "include_output_block": bool(args.include_output_block),
        "target": str(args.target),
        "rank_layers": {
            "mean_total_positive_local_minima_by_layer": mean_rank.tolist(),
            "std_total_positive_local_minima_by_layer": std_rank.tolist(),
        },
        "final_output": {
            "mean_total_positive_local_minima": float(np.mean(np.array(counts_output_by_run, dtype=float))),
            "std_total_positive_local_minima": float(np.std(np.array(counts_output_by_run, dtype=float))),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "positive_local_minima_per_run.json").open("w") as f:
        json.dump(per_run_records, f, indent=2)
    with (out_dir / "positive_local_minima_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_mean_with_std(
        mean_rank,
        std_rank,
        out_dir / "mean_positive_local_minima_vs_layer.png",
        title="mean strictly-positive local minima vs layer",
        xlabel="layer index",
        ylabel="mean # strictly-positive local minima per component",
    )
    if len(selected_runs) == 1:
        single = per_run_records[0]
        per_layer_arrays = [np.asarray(v, dtype=int) for v in single["layer_local_minima_per_component"]]
        plot_boxplot_from_arrays(
            per_layer_arrays,
            out_dir / "distribution_positive_local_minima_vs_layer_boxplot.png",
            title=f"distribution of minima across components vs layer (target={single['target']})",
            xlabel="layer index",
            ylabel="# strictly-positive local minima per component",
        )
        pooled = np.asarray(single["pooled_local_minima_per_component"], dtype=int)
        plot_histogram_from_values(
            pooled,
            out_dir / "hist_positive_local_minima_pooled_components.png",
            title=f"histogram of minima across all components (target={single['target']}, L={single['num_layers']})",
            xlabel="# strictly-positive local minima per component",
            ylabel="count of components",
        )

    if bool(args.leap_all_layers) and len(selected_runs) > 1:
        leap_records = [r.get("leap_all_layers", {}) for r in per_run_records]
        leap_records = [r for r in leap_records if isinstance(r, dict) and r.get("n_leap_layers", 0) > 0]
        if len(leap_records) > 0:
            means_by_run = [np.asarray(r["mean_minima_per_component_by_leap_layer"], dtype=float) for r in leap_records]
            min_len = int(min(m.size for m in means_by_run))
            means_by_run = [m[:min_len] for m in means_by_run]
            arr = np.stack(means_by_run, axis=0)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            pooled_all = np.concatenate([np.asarray(r.get("pooled_minima_per_component", []), dtype=int) for r in leap_records], axis=0)
            agg_dir = out_dir / "leap_all_layers_aggregate"
            plot_mean_curve(
                mean,
                agg_dir / "mean_minima_per_component_vs_layer_across_runs.png",
                title="mean minima per component (across runs)",
                xlabel="layer index",
                ylabel="mean minima per component",
            )
            plot_histogram_linear(
                pooled_all,
                agg_dir / "hist_minima_pooled_all_runs.png",
                title="pooled minima histogram across all runs/layers/components",
                xlabel="# strictly-positive local minima per component",
                ylabel="count of components (pooled)",
                bins=int(args.hist_bins),
            )
            with (agg_dir / "aggregate_summary.json").open("w") as f:
                json.dump(
                    {
                        "n_runs_used": int(len(leap_records)),
                        "common_leap_layers": int(min_len),
                        "value_threshold": float(args.value_threshold),
                        "hist_bins": int(args.hist_bins),
                    },
                    f,
                    indent=2,
                )

    if bool(args.leap_all_layers) and len(selected_runs) > 1 and str(args.group_by).strip() != "":
        group_fields = [p.strip().upper() for p in str(args.group_by).split(",") if p.strip() != ""]
        allowed = {"L", "W", "R"}
        if any(g not in allowed for g in group_fields):
            raise ValueError(f"unsupported group_by fields: {group_fields}; allowed {sorted(allowed)}")

        def group_key(rec: Dict[str, object]) -> str:
            parts: List[str] = []
            for g in group_fields:
                if g == "L":
                    parts.append(f"L{int(rec['num_layers'])}")
                elif g == "W":
                    parts.append(f"W{int(rec['hidden_width'])}")
                elif g == "R":
                    parts.append(f"R{int(rec['hidden_rank'])}")
            return "_".join(parts)

        grouped_dir = out_dir / "grouped"
        grouped_dir.mkdir(parents=True, exist_ok=True)

        groups: Dict[str, List[Dict[str, object]]] = {}
        for rec in per_run_records:
            if not isinstance(rec.get("leap_all_layers", {}), dict):
                continue
            if int(rec.get("leap_all_layers", {}).get("n_leap_layers", 0)) <= 0:
                continue
            groups.setdefault(group_key(rec), []).append(rec)

        csv_rows: List[Dict[str, object]] = []
        group_summary: Dict[str, object] = {"group_by": group_fields, "groups": {}}

        for gkey, recs in sorted(groups.items(), key=lambda kv: kv[0]):
            leap_recs = [r["leap_all_layers"] for r in recs]
            n_layers_common = int(min(int(r["n_leap_layers"]) for r in leap_recs))
            pooled_all = np.concatenate([np.asarray(r.get("pooled_minima_per_component", []), dtype=int) for r in leap_recs], axis=0)
            gout = grouped_dir / gkey
            if args.group_hist_mode in {"pooled", "both"}:
                plot_histogram_linear(
                    pooled_all,
                    gout / "hist_pooled_all_layers_components.png",
                    title=f"pooled minima histogram ({gkey})",
                    xlabel="# strictly-positive local minima per component",
                    ylabel="count of components (pooled)",
                    bins=int(args.hist_bins),
                )

            means_by_layer: List[float] = []
            stds_by_layer: List[float] = []
            for layer_idx in range(1, n_layers_common + 1):
                vals: List[int] = []
                for r in leap_recs:
                    per_layer = r.get("minima_per_component_by_leap_layer", [])
                    if isinstance(per_layer, list) and len(per_layer) >= layer_idx:
                        vals.extend([int(v) for v in per_layer[layer_idx - 1]])
                arr = np.asarray(vals, dtype=float)
                mu = float(np.mean(arr)) if arr.size > 0 else 0.0
                sig = float(np.std(arr)) if arr.size > 0 else 0.0
                means_by_layer.append(mu)
                stds_by_layer.append(sig)
                csv_rows.append(
                    {
                        "group": gkey,
                        "n_runs": int(len(leap_recs)),
                        "leap_layer_idx": int(layer_idx),
                        "mean_minima_per_component": mu,
                        "std_minima_per_component": sig,
                        "q25": float(np.quantile(arr, 0.25)) if arr.size > 0 else 0.0,
                        "q50": float(np.quantile(arr, 0.50)) if arr.size > 0 else 0.0,
                        "q75": float(np.quantile(arr, 0.75)) if arr.size > 0 else 0.0,
                        "n_components": int(arr.size),
                    }
                )
                if args.group_hist_mode in {"per_layer", "both"}:
                    plot_histogram_linear(
                        arr,
                        gout / f"hist_layer_{layer_idx:02d}.png",
                        title=f"minima histogram ({gkey}, layer {layer_idx})",
                        xlabel="# strictly-positive local minima per component",
                        ylabel="count of components (pooled over runs)",
                        bins=int(args.hist_bins),
                    )

            plot_mean_with_std(
                np.asarray(means_by_layer, dtype=float),
                np.asarray(stds_by_layer, dtype=float),
                gout / "mean_minima_per_component_vs_layer.png",
                title=f"{gkey}",
                xlabel="layer index",
                ylabel="mean minima per component",
            )
            group_summary["groups"][gkey] = {"n_runs": int(len(leap_recs)), "common_leap_layers": int(n_layers_common)}

        csv_path = grouped_dir / "group_layer_stats.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["group", "n_runs", "leap_layer_idx", "mean_minima_per_component", "std_minima_per_component", "q25", "q50", "q75", "n_components"],
            )
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        with (grouped_dir / "group_summary.json").open("w") as f:
            json.dump(group_summary, f, indent=2)

    print(f"wrote outputs to: {out_dir}")
    print(f"n_runs={len(selected_runs)}  layers_plotted={len(mean_rank)}  final_output_mean={summary['final_output']['mean_total_positive_local_minima']:.3f}")


if __name__ == "__main__":
    main()


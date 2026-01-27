#!/usr/bin/env python3
"""
Plot (1) the multi-frequency target used in frequency-layer scaling and
(2) the best test errors (down to ~1.4e-6) from those experiments.

Target: f(x) = cos(12f*pi*x) + cos(24f*pi*x+0.5) + cos(36f*pi*x) + cos(72f*pi*x+0.5)
on x in [-1,1], freq_multiplier f=0.3.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# we match train_frequency_layer_scaling target
def target_function(x: np.ndarray, freq_multiplier: float) -> np.ndarray:
    base_freqs = [12, 24, 36, 72]
    scaled = [f * freq_multiplier for f in base_freqs]
    return (
        np.cos(scaled[0] * np.pi * x)
        + np.cos(scaled[1] * np.pi * x + 0.5)
        + np.cos(scaled[2] * np.pi * x)
        + np.cos(scaled[3] * np.pi * x + 0.5)
    )


def parse_config_name(name: str) -> dict | None:
    """e.g. freq0.3_rank25_L5 -> {freq: 0.3, rank: 25, layers: 5}."""
    m = re.match(r"freq([\d.]+)_rank(\d+)_L(\d+)", name)
    if not m:
        return None
    return {"freq": float(m.group(1)), "rank": int(m.group(2)), "layers": int(m.group(3))}


def _load_results(results_dir: Path) -> list[dict]:
    """we load all configs with finite final_test_error."""
    out = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        rf = d / "results.json"
        if not rf.exists():
            continue
        try:
            with open(rf) as f:
                r = json.load(f)
            err = r.get("final_test_error")
            if err is None or (isinstance(err, float) and (np.isnan(err) or np.isinf(err))):
                continue
            err = float(err)
            if err > 1e10:  # skip blow-ups for display
                continue
        except Exception:
            continue
        parsed = parse_config_name(d.name)
        if parsed is None:
            continue
        out.append({
            "name": d.name,
            "freq": parsed["freq"],
            "rank": parsed["rank"],
            "layers": parsed["layers"],
            "final_test_error": err,
        })
    return out


def load_best_results(results_dir: Path, max_err: float = 1e-4) -> list[dict]:
    """we load configs with final_test_error <= max_err."""
    all_r = _load_results(results_dir)
    out = [x for x in all_r if x["final_test_error"] <= max_err]
    out.sort(key=lambda x: x["final_test_error"])
    return out


def load_bad_results(results_dir: Path, min_err: float = 0.5, max_count: int = 10) -> list[dict]:
    """we load configs with final_test_error >= min_err; pick several L so 'for several L training sucks'."""
    all_r = _load_results(results_dir)
    bad = [x for x in all_r if x["final_test_error"] >= min_err]
    bad.sort(key=lambda x: (x["layers"], -x["final_test_error"]))
    # one per L first for diversity, then fill
    by_L: dict[int, dict] = {}
    for b in bad:
        if b["layers"] not in by_L:
            by_L[b["layers"]] = b
    chosen = list(by_L.values())
    for b in bad:
        if len(chosen) >= max_count:
            break
        if b not in chosen:
            chosen.append(b)
    chosen.sort(key=lambda x: x["final_test_error"], reverse=True)
    return chosen[:max_count]


def _mmnn_param_count(hidden_width: int, hidden_rank: int, num_layers: int) -> int:
    """MMNN params: ranks [1,r,..,r,1], widths [W]*(num_layers+1). Matches train_frequency_layer_scaling."""
    first = 2 * hidden_width  # (1 -> W)
    block = num_layers * (2 * hidden_width * hidden_rank + hidden_width + hidden_rank)
    last = hidden_width + 1  # (W -> 1)
    return first + block + last


def setup_mpl():
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.rcParams["font.size"] = 14
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True


def main():
    base = Path(__file__).resolve().parent  # experiments/table/
    # we use the nested results path where the 1e-6 data lives
    results_dir = base / "experiments" / "table" / "results_frequency_layer_scaling"
    if not results_dir.exists():
        results_dir = base / "results_frequency_layer_scaling"
    out_dir = base / "experiments" / "table" if (base / "experiments" / "table").is_dir() else base
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_mpl()

    # ------ (A) Target function (f=0.6 only) ------
    x = np.linspace(-1, 1, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    f = 0.6
    y = target_function(x, f)
    axes[0].plot(x, y, color="#1f77b4", alpha=0.9, linewidth=1.2)
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_title(r"Target: $\sum_{i=1}^4 \cos(k_i f\pi x+\phi_i)$, $k{=}(12,24,36,72)$, $f{=}0.6$")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-1, 1)

    # ------ (B) Good (low err) + bad (high err), only f=0.3 and f=0.6 ------
    best = load_best_results(results_dir, max_err=1e-4)
    if not best:
        best = load_best_results(results_dir, max_err=3e-2)
    bad = load_bad_results(results_dir, min_err=0.5, max_count=10)
    best = [b for b in best if b["freq"] in (0.3, 0.6)]
    bad = [b for b in bad if b["freq"] in (0.3, 0.6)]
    combined = best + bad

    HIDDEN_WIDTH = 777  # from train_frequency_layer_scaling
    labels = [f"f={b['freq']} r={b['rank']} L={b['layers']}" for b in combined]
    # we add parameter count only for the first r=10 L=3 bar
    for i, b in enumerate(combined):
        if b["rank"] == 10 and b["layers"] == 3:
            n = _mmnn_param_count(HIDDEN_WIDTH, 10, 3)
            labels[i] += f" ({n//1000}k)"
            break
    errs = [b["final_test_error"] for b in combined]
    n_good = len(best)
    colors = (
        ["#2ecc71" if e < 1e-5 else "#3498db" for e in errs[:n_good]]
        + ["#e74c3c" for _ in errs[n_good:]]
    )

    ax = axes[1]
    ypos = np.arange(len(combined), 0, -1)
    ax.barh(ypos, errs, color=colors, alpha=0.85, edgecolor="none")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Final test error (MSE)")
    ax.set_xscale("log")
    ax.set_title("Depth/rank training comparisons")
    ax.axvline(1e-6, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="$10^{-6}$")
    ax.axvline(1e-5, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="$10^{-5}$")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(left=min(errs) / 2, right=1e-4)

    plt.tight_layout()
    path = out_dir / "target_and_astonishing_results.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # ------ Optional: target-only figure (single function) ------
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    f = 0.3
    ax.plot(x, target_function(x, f), color="#1f77b4", alpha=0.9, linewidth=1.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"Target $f(x)=\sum_{i=1}^4 \cos(k_i f\pi x + \phi_i)$ on $[-1,1]$, $f{=}0.3$")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    path2 = out_dir / "target_multi_frequency.png"
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"Saved {path2}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
we plot partial layer functions for all saved model_parameters in the frequency benchmark
same style as experiments/former/SinQuad/leap.py (layer-wise component plots)
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.table.mmnn_vs import MMNN


# we limit components per layer for readable plots (as in leap.py)
MAX_COMPONENTS = 36


def load_model_state(dirpath: Path, device: torch.device):
    """we load model state from model_parameters.pth or checkpoint.pth"""
    model_path = dirpath / "model_parameters.pth"
    checkpoint_path = dirpath / "checkpoint.pth"
    if model_path.exists():
        return torch.load(model_path, map_location=device)
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        return ckpt.get("model_state_dict", ckpt)
    return None


def plot_partials_for_config(config_dir: Path, device: torch.device, out_subdir: str = "partials"):
    """we load model, compute layer outputs, and plot partial functions for one config"""
    config_path = config_dir / "config.json"
    if not config_path.exists():
        print(f"skip {config_dir.name}: no config.json")
        return False

    with open(config_path) as f:
        cfg = json.load(f)

    state = load_model_state(config_dir, device)
    if state is None:
        print(f"skip {config_dir.name}: no model_parameters.pth or checkpoint.pth")
        return False

    num_layers = int(cfg["num_layers"])
    hidden_width = int(cfg["hidden_width"])
    hidden_rank = int(cfg["hidden_rank"])
    input_rank = int(cfg.get("input_rank", 1))
    output_rank = int(cfg.get("output_rank", 1))
    use_resnet = bool(cfg.get("use_resnet", False))
    fix_wb = bool(cfg.get("fixWb", False))
    interval = cfg.get("interval", [-1, 1])
    freq1 = cfg.get("freq1", 36)
    freq2 = cfg.get("freq2", 12)

    ranks = [input_rank] + [hidden_rank] * num_layers + [output_rank]
    widths = [hidden_width] * (num_layers + 1)

    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=device,
        ResNet=use_resnet,
        fixWb=fix_wb,
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    x = np.linspace(interval[0], interval[1], 1000)
    x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.get_default_dtype(), device=device)

    # we compute layer outputs (after each fc; after relu for even indices) as in leap.py
    layer_outputs = {}
    with torch.no_grad():
        current = x_tensor
        for i in range(len(model.fcs)):
            current = model.fcs[i](current)
            if i % 2 == 0:
                current = torch.relu(current)
            layer_outputs[i] = current.cpu().numpy()

    # we create output subdir for partials
    partials_dir = config_dir / out_subdir
    partials_dir.mkdir(parents=True, exist_ok=True)

    config_str = f"freq=({freq1},{freq2}) rank={hidden_rank} fixWb={fix_wb}"

    # we plot each layer (1 to len(fcs)-1) as in leap.py
    for layer_idx in range(1, len(model.fcs)):
        out = layer_outputs[layer_idx]
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        n_components = out.shape[1]
        n_plot = min(n_components, MAX_COMPONENTS)

        n_rows = int(np.ceil(np.sqrt(n_plot)))
        n_cols = int(np.ceil(n_plot / n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.array(axes).reshape(n_rows, n_cols)

        fig.suptitle(
            f"Partial functions – Layer {layer_idx} ({n_plot} components)\n{config_str}",
            fontsize=14,
        )

        for idx in range(n_plot):
            i, j = idx // n_cols, idx % n_cols
            axes[i, j].plot(x, out[:, idx], "b-", linewidth=1)
            axes[i, j].set_title(f"Component {idx + 1}")
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].set_xticks([interval[0], 0, interval[1]])

        # we hide unused subplots
        for idx in range(n_plot, n_rows * n_cols):
            i, j = idx // n_cols, idx % n_cols
            axes[i, j].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_file = partials_dir / f"layer_{layer_idx}_components.png"
        plt.savefig(out_file, dpi=100)
        plt.close()

    print(f"plotted partials: {config_dir.name} -> {partials_dir}")
    return True


def main():
    base = Path(__file__).parent
    # we search in both possible result locations
    candidates = [
        base / "experiments" / "table" / "results_frequency_benchmark",
        base / "results_frequency_benchmark",
    ]
    results_dir = None
    for d in candidates:
        if d.is_dir():
            results_dir = d
            break
    if results_dir is None:
        print("no results_frequency_benchmark directory found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # we collect all config dirs that have config.json
    config_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir() and (p / "config.json").exists()])
    # we only process dirs that have a saved model
    model_dirs = [d for d in config_dirs if (d / "model_parameters.pth").exists() or (d / "checkpoint.pth").exists()]

    print(f"found {len(model_dirs)} configs with saved models in {results_dir}")
    print(f"processing {len(model_dirs)} configs...\n")

    success_count = 0
    for idx, d in enumerate(model_dirs, 1):
        print(f"[{idx}/{len(model_dirs)}] processing {d.name}...", end=" ", flush=True)
        try:
            if plot_partials_for_config(d, device, out_subdir="partials"):
                success_count += 1
                print("✓")
            else:
                print("✗ (skipped)")
        except Exception as e:
            print(f"✗ error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\ndone. successfully processed {success_count}/{len(model_dirs)} configs")


if __name__ == "__main__":
    main()

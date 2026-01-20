import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # we use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class MMNN(nn.Module):
    def __init__(
        self,
        ranks: List[int],
        widths: List[int],
        device: str = "cpu",
        ResNet: bool = False,
        fixWb: bool = False,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.product = 1.0  # we store the normalization product
        for j in range(1, len(ranks)):
            self.product *= float(np.sqrt(widths[j - 1] * ranks[j]))  # we match the training normalization
        self.ranks = ranks  # we store ranks
        self.widths = widths  # we store widths
        self.ResNet = ResNet  # we store resnet flag
        self.depth = len(widths)  # we store number of blocks
        self.normalize_output = bool(normalize_output)  # we store output normalization flag

        fc_sizes = [ranks[0]]  # we build layer sizes
        for j in range(self.depth):
            fc_sizes += [widths[j], ranks[j + 1]]  # we append width and rank

        fcs: List[nn.Linear] = []  # we store linear layers
        for j in range(len(fc_sizes) - 1):
            fc = nn.Linear(fc_sizes[j], fc_sizes[j + 1], device=device)  # we create linear layer
            fcs.append(fc)  # we append layer
        self.fcs = nn.ModuleList(fcs)  # we store layers

        if fixWb:
            for j in range(len(fcs)):
                if j % 2 == 0:
                    self.fcs[j].weight.requires_grad = False  # we freeze weight
                    self.fcs[j].bias.requires_grad = False  # we freeze bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for j in range(self.depth):
            if self.ResNet and 0 < j < self.depth - 1:
                x_id = x + 0  # we store identity connection
            x = self.fcs[2 * j](x)  # we apply fixed part
            x = torch.relu(x)  # we apply relu
            x = self.fcs[2 * j + 1](x)  # we apply low-rank part
            if self.ResNet and 0 < j < self.depth - 1:
                n = min(x.shape[1], x_id.shape[1])  # we match ranks
                x[:, :n] = x[:, :n] + x_id[:, :n]  # we add residual
        if self.normalize_output:
            return x / float(self.product)  # we normalize output
        return x  # we return unnormalized output


def windows_to_wsl_path(path: str) -> str:
    m = re.match(r"^([A-Za-z]):\\(.*)$", path)  # we match windows drive path
    if not m:
        return path  # we keep original path
    drive = m.group(1).lower()  # we take drive letter
    rest = m.group(2).replace("\\", "/")  # we convert separators
    return f"/mnt/{drive}/{rest}"  # we map to wsl mount


def resolve_checkpoint_path(path: str) -> str:
    candidates = [path, windows_to_wsl_path(path)]  # we try both representations
    for p in candidates:
        if os.path.exists(p):
            return p  # we return existing path
    raise FileNotFoundError(f"checkpoint not found: tried {candidates}")  # we raise if missing


@dataclass(frozen=True)
class ArchConfig:
    num_layers: int
    hidden_width: int
    hidden_rank: int
    input_rank: int = 1
    output_rank: int = 1
    use_resnet: bool = False
    normalize_output: bool = True


def parse_arch_from_folder_name(folder_name: str) -> ArchConfig:
    m = re.search(r"_L(\d+)_W(\d+)_R(\d+)_", folder_name)  # we parse the common run naming
    if m is None:
        raise ValueError(f"could not parse arch from folder name: {folder_name}")  # we raise on failure
    return ArchConfig(
        num_layers=int(m.group(1)),
        hidden_width=int(m.group(2)),
        hidden_rank=int(m.group(3)),
    )


def load_arch_from_config_json(folder: str) -> Optional[ArchConfig]:
    config_path = os.path.join(folder, "config.json")  # we set config path
    if not os.path.exists(config_path):
        return None  # we return none if missing
    with open(config_path, "r") as f:
        cfg = json.load(f)  # we load config json
    return ArchConfig(
        num_layers=int(cfg.get("num_layers", cfg.get("L", 0))),
        hidden_width=int(cfg.get("hidden_width", cfg.get("W", 0))),
        hidden_rank=int(cfg.get("hidden_rank", cfg.get("R", 0))),
        input_rank=int(cfg.get("input_rank", 1)),
        output_rank=int(cfg.get("output_rank", 1)),
        use_resnet=bool(cfg.get("use_resnet", cfg.get("ResNet", False))),
        normalize_output=bool(cfg.get("normalize_output", True)),
    )


def build_mmnn_from_arch(arch: ArchConfig) -> MMNN:
    ranks = [arch.input_rank] + [arch.hidden_rank] * arch.num_layers + [arch.output_rank]  # we build ranks
    widths = [arch.hidden_width] * (arch.num_layers + 1)  # we build widths like training scripts
    return MMNN(
        ranks=ranks,
        widths=widths,
        device="cpu",
        ResNet=arch.use_resnet,
        fixWb=False,
        normalize_output=arch.normalize_output,
    )


def resolve_device(device_str: str) -> torch.device:
    s = device_str.strip().lower()  # we normalize input
    if s in ["cuda", "cuda:0"]:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # we default to cuda if available
    if s.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(device_str)  # we use requested cuda device
        return torch.device("cpu")  # we fall back to cpu
    return torch.device("cpu")  # we default to cpu


def layer_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "count": float(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "l2_norm": float(np.linalg.norm(x.reshape([-1]))),
    }


def plot_hist(data: np.ndarray, title: str, out_path: str, bins: int = 200) -> None:
    fig = plt.figure(figsize=(8, 4))  # we create a figure
    plt.hist(data.reshape([-1]), bins=bins, density=True, alpha=0.75, color="C0")  # we plot histogram
    plt.grid(True, alpha=0.3)  # we add grid
    plt.title(title)  # we set title
    plt.tight_layout()  # we layout
    plt.savefig(out_path, dpi=140)  # we save figure
    plt.close(fig)  # we close figure


def is_trained_layer_idx(i: int) -> bool:
    return (i % 2) == 1  # we treat odd indices as trained low-rank layers


def iter_checkpoints(root_dir: str, filename: str = "model_parameters.pth") -> Iterable[str]:
    for root, _, files in os.walk(root_dir):
        if filename in files:
            yield os.path.join(root, filename)  # we yield checkpoint path


def process_checkpoint(
    ckpt_path: str,
    device: torch.device,
    bins: int,
    layers_mode: str,
    out_mode: str,
    out_dir: str,
    out_subdir: str,
) -> str:
    ckpt_dir = os.path.dirname(ckpt_path)  # we locate checkpoint folder
    ckpt_name = os.path.basename(ckpt_dir)  # we read run folder name
    ckpt_hash = hashlib.md5(ckpt_dir.encode("utf-8")).hexdigest()[:10]  # we create a short stable id

    arch = load_arch_from_config_json(ckpt_dir)  # we try to load config.json
    if arch is None:
        arch = parse_arch_from_folder_name(ckpt_name)  # we parse from folder name

    out_dirs: List[str] = []  # we store output directories
    if out_mode in ["plots", "both"]:
        plots_out = os.path.join(out_dir, f"{ckpt_name}__{ckpt_hash}")  # we create a unique folder per checkpoint
        os.makedirs(plots_out, exist_ok=True)  # we create plots output directory
        out_dirs.append(plots_out)  # we store plots output directory
    if out_mode in ["inplace", "both"]:
        inplace_out = os.path.join(ckpt_dir, out_subdir)  # we save next to the checkpoint
        os.makedirs(inplace_out, exist_ok=True)  # we create inplace output directory
        out_dirs.append(inplace_out)  # we store inplace output directory
    if len(out_dirs) == 0:
        raise ValueError(f"invalid out_mode={out_mode}")  # we validate output mode

    model = build_mmnn_from_arch(arch).to(device)  # we build the model on device
    state = torch.load(ckpt_path, map_location="cpu")  # we load state dict on cpu
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]  # we unwrap state dict if wrapped
    if not isinstance(state, dict):
        raise TypeError(f"unexpected checkpoint type: {type(state)}")  # we raise on unexpected type
    missing, unexpected = model.load_state_dict(state, strict=False)  # we load parameters

    index_base: Dict[str, object] = {
        "checkpoint_path": ckpt_path,
        "checkpoint_dir": ckpt_dir,
        "arch": arch.__dict__,
        "device": str(device),
        "layers_mode": layers_mode,
        "out_mode": out_mode,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "layers": [],
    }  # we build output index

    all_weights = []  # we collect all weights
    all_biases = []  # we collect all biases

    for i, layer in enumerate(model.fcs):
        if layers_mode == "trained" and (not is_trained_layer_idx(i)):
            continue  # we skip frozen layers
        w = layer.weight.detach().cpu().numpy()  # we get weight
        b = layer.bias.detach().cpu().numpy()  # we get bias
        all_weights.append(w.reshape([-1]))  # we accumulate weights
        all_biases.append(b.reshape([-1]))  # we accumulate biases

        for out_root in out_dirs:
            layer_dir = os.path.join(out_root, f"layer_{i:02d}")  # we create layer directory
            os.makedirs(layer_dir, exist_ok=True)  # we create directory
            plot_hist(w, title=f"{ckpt_name} | layer {i} weight histogram", out_path=os.path.join(layer_dir, "weight_hist.png"), bins=bins)  # we plot weight hist
            plot_hist(b, title=f"{ckpt_name} | layer {i} bias histogram", out_path=os.path.join(layer_dir, "bias_hist.png"), bins=bins)  # we plot bias hist

        idx_layer = {
            "layer_idx": i,
            "trained_layer": bool(is_trained_layer_idx(i)),
            "weight_shape": list(w.shape),
            "bias_shape": list(b.shape),
            "weight_stats": layer_stats(w),
            "bias_stats": layer_stats(b),
        }  # we store layer stats
        index_base["layers"].append(idx_layer)  # we append layer index

    w_all = np.concatenate(all_weights, axis=0) if len(all_weights) > 0 else np.zeros([0])  # we concatenate weights
    b_all = np.concatenate(all_biases, axis=0) if len(all_biases) > 0 else np.zeros([0])  # we concatenate biases
    if w_all.size > 0:
        for out_root in out_dirs:
            plot_hist(w_all, title=f"{ckpt_name} | all selected layers weight histogram", out_path=os.path.join(out_root, "all_weights_hist.png"), bins=bins)  # we plot all weights
            plot_hist(b_all, title=f"{ckpt_name} | all selected layers bias histogram", out_path=os.path.join(out_root, "all_biases_hist.png"), bins=bins)  # we plot all biases
        index_base["all_weight_stats"] = layer_stats(w_all)  # we store global weight stats
        index_base["all_bias_stats"] = layer_stats(b_all)  # we store global bias stats

    for out_root in out_dirs:
        index = dict(index_base)  # we create a per-output index
        index["out_dir"] = out_root  # we store output directory
        with open(os.path.join(out_root, "index.json"), "w") as f:
            json.dump(index, f, indent=2)  # we save index json
        print(f"saved plots to: {out_root}")  # we print output path
    return out_dirs[0]  # we return primary output dir


def main() -> None:
    parser = argparse.ArgumentParser()  # we parse cli args
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"G:\Mon Drive\JANIS AIAD Internship - NTK for NN\mmnn_training\mmnn_L6_W1024_R15_E3000_lr0.001_bs100_ntr1000\model_parameters.pth",
    )  # we set default checkpoint path
    parser.add_argument("--root_dir", type=str, default="")  # we optionally scan a root directory
    parser.add_argument("--out_dir", type=str, default="/home/janis/STG3A/MMNN/experiments/investigate_weights/plots")  # we set base output directory for plots mode
    parser.add_argument("--bins", type=int, default=200)  # we set histogram bins
    parser.add_argument("--device", type=str, default="cuda:0")  # we choose device for loading
    parser.add_argument("--layers_mode", type=str, default="trained", choices=["trained", "all"])  # we select layers to plot
    parser.add_argument("--out_mode", type=str, default="plots", choices=["plots", "inplace", "both"])  # we choose where to write outputs
    parser.add_argument("--out_subdir", type=str, default="weight_distributions")  # we choose output subdir name for inplace mode
    parser.add_argument("--max_models", type=int, default=0)  # we cap number of models if positive
    args = parser.parse_args()  # we parse args

    device = resolve_device(str(args.device))  # we resolve device
    if str(args.root_dir).strip():
        root_dir = resolve_checkpoint_path(str(args.root_dir))  # we resolve root dir path (supports windows drive form)
        paths = list(iter_checkpoints(root_dir, filename="model_parameters.pth"))  # we list checkpoints
        paths = sorted(paths)  # we sort for deterministic order
        if int(args.max_models) > 0:
            paths = paths[: int(args.max_models)]  # we cap
        for p in paths:
            process_checkpoint(
                p,
                device=device,
                bins=int(args.bins),
                layers_mode=str(args.layers_mode),
                out_mode=str(args.out_mode),
                out_dir=str(args.out_dir),
                out_subdir=str(args.out_subdir),
            )  # we process each
        print(f"processed_models: {len(paths)}")  # we print count
        return

    ckpt_path = resolve_checkpoint_path(str(args.checkpoint))  # we resolve checkpoint path
    process_checkpoint(
        ckpt_path,
        device=device,
        bins=int(args.bins),
        layers_mode=str(args.layers_mode),
        out_mode=str(args.out_mode),
        out_dir=str(args.out_dir),
        out_subdir=str(args.out_subdir),
    )  # we process one


if __name__ == "__main__":
    main()

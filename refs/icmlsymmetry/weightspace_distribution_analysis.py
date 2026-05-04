#!/usr/bin/env python3
"""Plot weight-space distributions for symmetric and asymmetric MMNN runs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from refs.icmlsymmetry.good_sumcos_low_loss import load_model  # noqa: E402


SOURCE_ROOT = Path(__file__).resolve().parent / "results" / "all_iclr_sumcos_rerun"
OUT_ROOT = Path(__file__).resolve().parent / "results" / "weightspace_distributions"
EPS = 1e-12


def read_summary(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def finite_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value


def classify(row: dict[str, str], loss_threshold: float, symmetric_threshold: float, asymmetric_threshold: float) -> str:
    loss = finite_float(row, "final_test_error")
    partial = finite_float(row, "last_layer_active_even_mean")
    if not np.isfinite(loss) or not np.isfinite(partial):
        return "unknown"
    if loss > max(1e-2, 10.0 * loss_threshold):
        return "underfit"
    if loss <= loss_threshold and partial <= symmetric_threshold:
        return "partial-symmetric"
    if loss <= loss_threshold and partial >= asymmetric_threshold:
        return "output-only/asymmetric"
    return "intermediate"


def load_run_model(run_name: str) -> tuple[object, dict] | None:
    run_dir = SOURCE_ROOT / run_name
    losses_path = run_dir / "losses.json"
    params_path = run_dir / "model_parameters.pth"
    if not losses_path.exists() or not params_path.exists():
        return None
    with open(losses_path) as f:
        losses = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(losses["config"], params_path, device)
    return model, losses


def first_layer_arrays(model: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = model.fcs[0]
    second = model.fcs[1]
    slopes = first.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    biases = first.bias.detach().cpu().numpy().reshape(-1).astype(np.float64)
    outgoing = second.weight.detach().cpu().numpy().astype(np.float64)
    return slopes, biases, outgoing


def mirror_pair_distribution(model: object) -> dict[str, np.ndarray]:
    slopes, biases, outgoing = first_layer_arrays(model)
    n_atoms = slopes.shape[0]
    partner = np.zeros(n_atoms, dtype=np.int64)
    distance = np.zeros(n_atoms, dtype=np.float64)
    mismatch = np.zeros(n_atoms, dtype=np.float64)
    correlation = np.zeros(n_atoms, dtype=np.float64)
    for j in range(n_atoms):
        squared = (slopes + slopes[j]) ** 2 + (biases - biases[j]) ** 2
        squared[j] = np.inf
        k = int(np.argmin(squared))
        cj = outgoing[:, j]
        ck = outgoing[:, k]
        denom = float(np.mean(cj ** 2 + ck ** 2) + EPS)
        partner[j] = k
        distance[j] = float(np.sqrt(squared[k]))
        mismatch[j] = float(np.mean((cj - ck) ** 2) / denom)
        correlation[j] = float(np.dot(cj, ck) / (np.linalg.norm(cj) * np.linalg.norm(ck) + EPS))
    return {
        "slopes": slopes,
        "biases": biases,
        "outgoing_abs": np.abs(outgoing).reshape(-1),
        "partner": partner,
        "mirror_distance": distance,
        "outgoing_mismatch": mismatch,
        "outgoing_correlation": correlation,
    }


def choose_examples(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    valid = [
        row for row in rows
        if np.isfinite(finite_float(row, "final_test_error"))
        and np.isfinite(finite_float(row, "last_layer_active_even_mean"))
    ]
    symmetric = sorted(valid, key=lambda row: (finite_float(row, "last_layer_active_even_mean"), finite_float(row, "final_test_error")))
    asymmetric = sorted(
        [row for row in valid if finite_float(row, "final_test_error") <= 1e-3],
        key=lambda row: finite_float(row, "last_layer_active_even_mean"),
        reverse=True,
    )
    underfit = sorted(valid, key=lambda row: finite_float(row, "final_test_error"), reverse=True)
    examples: dict[str, dict[str, str]] = {}
    if symmetric:
        examples["partial-symmetric"] = symmetric[0]
    if asymmetric:
        examples["output-only/asymmetric"] = asymmetric[0]
    if underfit:
        examples["underfit"] = underfit[0]
    return examples


def write_classification(rows: list[dict[str, str]], class_rows: list[dict[str, object]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "class", "factor", "rank", "N", "bs", "L", "final_test_error",
        "output_even_defect", "last_layer_active_even_mean", "mirror_distance_median",
        "mirror_mismatch_median", "mirror_corr_median",
    ]
    with open(OUT_ROOT / "weightspace_classification.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(class_rows)
    with open(OUT_ROOT / "weightspace_classification.json", "w") as f:
        json.dump(class_rows, f, indent=2)


def plot_global_distributions(class_arrays: dict[str, dict[str, list[np.ndarray]]]) -> None:
    styles = {
        "partial-symmetric": "#1b9e77",
        "output-only/asymmetric": "#d95f02",
        "underfit": "#7570b3",
        "intermediate": "#666666",
    }
    metrics = [
        ("mirror_distance", "log10 nearest mirror distance", "mirror_distance_distribution.png"),
        ("outgoing_mismatch", "log10 outgoing mirror mismatch", "outgoing_mismatch_distribution.png"),
        ("outgoing_correlation", "outgoing mirror correlation", "outgoing_correlation_distribution.png"),
        ("outgoing_abs", "log10 absolute outgoing coefficient", "outgoing_weight_distribution.png"),
    ]
    for key, xlabel, filename in metrics:
        plt.figure(figsize=(8, 5))
        for label, color in styles.items():
            values_list = class_arrays.get(label, {}).get(key, [])
            if not values_list:
                continue
            values = np.concatenate(values_list)
            if key != "outgoing_correlation":
                values = np.log10(values + EPS)
            plt.hist(values, bins=45, density=True, histtype="step", linewidth=2.0, color=color, label=label)
        plt.xlabel(xlabel)
        plt.ylabel("density")
        plt.title(xlabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_ROOT / filename, dpi=240)
        plt.close()


def plot_summary_scatter(class_rows: list[dict[str, object]]) -> None:
    rows = [
        row for row in class_rows
        if np.isfinite(float(row["final_test_error"]))
        and np.isfinite(float(row["last_layer_active_even_mean"]))
        and np.isfinite(float(row["mirror_mismatch_median"]))
    ]
    if not rows:
        return
    plt.figure(figsize=(8, 5.5))
    x = np.asarray([float(row["final_test_error"]) for row in rows])
    y = np.asarray([float(row["last_layer_active_even_mean"]) for row in rows])
    c = np.asarray([float(row["mirror_mismatch_median"]) for row in rows])
    sc = plt.scatter(x, y, c=np.log10(c + EPS), cmap="viridis", s=55, alpha=0.85)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("final test MSE")
    plt.ylabel("last-layer active partial even defect")
    plt.title("Function-space symmetry versus mirror mismatch")
    plt.colorbar(sc, label="log10 median outgoing mirror mismatch")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "loss_partial_vs_mirror_mismatch.png", dpi=240)
    plt.close()
    plt.figure(figsize=(8, 5.5))
    c_corr = np.asarray([float(row["mirror_corr_median"]) for row in rows])
    sc = plt.scatter(x, y, c=c_corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, s=55, alpha=0.85)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("final test MSE")
    plt.ylabel("last-layer active partial even defect")
    plt.title("Function-space symmetry versus mirror correlation")
    plt.colorbar(sc, label="median outgoing mirror correlation")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "loss_partial_vs_mirror_correlation.png", dpi=240)
    plt.close()


def plot_example(row: dict[str, str], arrays: dict[str, np.ndarray], label: str) -> None:
    name = row["name"]
    slopes = arrays["slopes"]
    biases = arrays["biases"]
    partner = arrays["partner"].astype(np.int64)
    mismatch = arrays["outgoing_mismatch"]
    order = np.argsort(mismatch)[:80]
    plt.figure(figsize=(7, 6))
    plt.scatter(slopes, biases, c=np.log10(mismatch + EPS), cmap="viridis", s=16, alpha=0.75)
    for j in order[::2]:
        k = int(partner[j])
        plt.plot([slopes[j], slopes[k]], [biases[j], biases[k]], color="black", alpha=0.12, linewidth=0.8)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
    plt.xlabel("first-layer slope")
    plt.ylabel("first-layer bias")
    plt.title(f"{label}: mirror pairs in first-layer atoms\n{name}")
    plt.colorbar(label="log10 outgoing mismatch")
    plt.tight_layout()
    safe_label = label.replace("/", "_").replace(" ", "_")
    plt.savefig(OUT_ROOT / f"example_{safe_label}_mirror_pairs.png", dpi=260)
    plt.close()
    plt.figure(figsize=(8, 4.5))
    plt.hist(np.log10(arrays["mirror_distance"] + EPS), bins=45, alpha=0.65, label="mirror distance")
    plt.hist(np.log10(mismatch + EPS), bins=45, alpha=0.65, label="outgoing mismatch")
    plt.xlabel("log10 value")
    plt.ylabel("count")
    plt.title(f"{label}: distance and mismatch\n{name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"example_{safe_label}_distance_mismatch_hist.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SOURCE_ROOT / "summary.csv")
    parser.add_argument("--loss-threshold", type=float, default=1e-3)
    parser.add_argument("--symmetric-threshold", type=float, default=1e-3)
    parser.add_argument("--asymmetric-threshold", type=float, default=1e-2)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_summary(args.summary)
    if args.max_runs is not None:
        rows = rows[: args.max_runs]
    examples = choose_examples(rows)
    class_arrays: dict[str, dict[str, list[np.ndarray]]] = {}
    class_rows: list[dict[str, object]] = []
    example_arrays: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        name = row["name"]
        loaded = load_run_model(name)
        if loaded is None:
            continue
        model, _losses = loaded
        arrays = mirror_pair_distribution(model)
        label = classify(row, args.loss_threshold, args.symmetric_threshold, args.asymmetric_threshold)
        class_arrays.setdefault(label, {})
        for key, values in arrays.items():
            if key == "partner":
                continue
            class_arrays[label].setdefault(key, []).append(values)
        class_rows.append({
            "name": name,
            "class": label,
            "factor": row.get("factor", ""),
            "rank": row.get("rank", ""),
            "N": row.get("N", ""),
            "bs": row.get("bs", ""),
            "L": row.get("L", ""),
            "final_test_error": finite_float(row, "final_test_error"),
            "output_even_defect": finite_float(row, "output_even_defect"),
            "last_layer_active_even_mean": finite_float(row, "last_layer_active_even_mean"),
            "mirror_distance_median": float(np.median(arrays["mirror_distance"])),
            "mirror_mismatch_median": float(np.median(arrays["outgoing_mismatch"])),
            "mirror_corr_median": float(np.median(arrays["outgoing_correlation"])),
        })
        for example_label, example_row in examples.items():
            if example_row["name"] == name:
                example_arrays[example_label] = arrays
    write_classification(rows, class_rows)
    plot_global_distributions(class_arrays)
    plot_summary_scatter(class_rows)
    for label, row in examples.items():
        arrays = example_arrays.get(label)
        if arrays is not None:
            plot_example(row, arrays, label)
    counts: dict[str, int] = {}
    for row in class_rows:
        label = str(row["class"])
        counts[label] = counts.get(label, 0) + 1
    print(f"analyzed {len(class_rows)} checkpoints -> {OUT_ROOT}")
    print(json.dumps(counts, indent=2))
    print("examples:")
    for label, row in examples.items():
        print(f"  {label}: {row['name']} loss={row['final_test_error']} partial={row['last_layer_active_even_mean']}")


if __name__ == "__main__":
    main()

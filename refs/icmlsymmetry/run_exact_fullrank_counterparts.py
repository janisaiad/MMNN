#!/usr/bin/env python3
"""Train one full-rank MLP counterpart for every completed MMNN multidim run."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refs.icmlsymmetry.multidim_symmetry_batch1 import Config, OUT_ROOT, train_one


def read_mmnn_rows() -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(OUT_ROOT.glob("mmnn_*/metrics.json")):
        with open(metrics_path) as f:
            row = json.load(f)
        if row.get("model") == "mmnn":
            rows.append(row)
    return rows


def counterpart_config(row: dict, epochs_override: int | None) -> Config:
    return Config(
        dim=int(row["dim"]),
        target=str(row["target"]),
        model="mlp",
        depth=int(row["depth"]),
        width=int(row["width"]),
        rank=0,
        freq=int(row["freq"]),
        n_train=int(row["n_train"]),
        batch_size=int(row["batch_size"]),
        epochs=int(epochs_override if epochs_override is not None else row["epochs"]),
        lr=float(row["lr"]),
        seed=int(row["seed"]),
        train_distribution=str(row.get("train_distribution", "unpaired_uniform")),
        optimizer=str(row.get("optimizer", "adam")),
    )


def write_counterpart_summary(rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with open(OUT_ROOT / "fullrank_counterparts_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_ROOT / "fullrank_counterparts_summary.json", "w") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    mmnn_rows = read_mmnn_rows()
    configs_by_name: dict[str, Config] = {}
    for row in mmnn_rows:
        config = counterpart_config(row, args.epochs_override)
        configs_by_name[config.name] = config
    configs = list(configs_by_name.values())
    if args.max_runs is not None:
        configs = configs[: args.max_runs]
    print(f"selected {len(configs)} exact full-rank counterparts", flush=True)
    rows: list[dict] = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.name}", flush=True)
        row = train_one(config, overwrite=args.overwrite)
        rows.append(row)
        write_counterpart_summary(rows)
    write_counterpart_summary(rows)
    print(f"done -> {OUT_ROOT / 'fullrank_counterparts_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

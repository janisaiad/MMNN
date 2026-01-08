import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys

_ROOT = Path(__file__).resolve().parents[2]  # we locate repo root #
if str(_ROOT) not in sys.path:  # we ensure imports from repo root #
    sys.path.insert(0, str(_ROOT))  # we add path #


@dataclass(frozen=True)
class RunSummary:  # we store run summary #
    run_dir: str  # we store path #
    model_type: str  # we store #
    branch_type: str  # we store #
    trunk_fix_wb: bool  # we store #
    branch_fix_wb: bool  # we store #
    trunk_rank: int  # we store #
    branch_rank: int  # we store #
    final_test_loss: float  # we store #
    final_test_rel_l2: float  # we store #


def _read_json(path: Path) -> dict:  # we read json #
    with open(path, "r") as f:  # we open #
        return json.load(f)  # we return #


def _read_metrics(metrics_path: Path) -> dict:  # we read last metrics row #
    last = None  # we init #
    with open(metrics_path, "r") as f:  # we open #
        for line in f:  # we iterate #
            line = line.strip()  # we strip #
            if not line:  # we skip #
                continue  # we continue #
            last = json.loads(line)  # we parse #
    if last is None:  # we guard #
        raise ValueError(f"no rows in {metrics_path}")  # we raise #
    return last  # we return #


def collect_runs(runs_dir: Path) -> list[RunSummary]:  # we collect runs #
    runs_dir = Path(runs_dir)  # we normalize #
    out = []  # we collect #
    for run in sorted(runs_dir.glob("*")):  # we iterate #
        if not run.is_dir():  # we skip #
            continue  # we continue #
        metrics = run / "metrics.jsonl"  # we set #
        model_cfg_p = run / "model_config.json"  # we set #
        if not metrics.exists() or not model_cfg_p.exists():  # we skip #
            continue  # we continue #
        mc = _read_json(model_cfg_p)  # we read #
        last = _read_metrics(metrics)  # we read #
        out.append(
            RunSummary(
                run_dir=str(run),
                model_type=str(mc.get("model_type", "")),
                branch_type=str(mc.get("branch_type", "")),
                trunk_fix_wb=bool(mc.get("trunk_fix_wb", False)),
                branch_fix_wb=bool(mc.get("branch_fix_wb", False)),
                trunk_rank=int(mc.get("trunk_rank", 0)),
                branch_rank=int(mc.get("branch_mmnn_rank", 0)),
                final_test_loss=float(last["test_loss"]),
                final_test_rel_l2=float(last["test_rel_l2"]),
            )
        )  # we append #
    return out  # we return #


def write_summary(runs_dir: Path, out_dir: Path) -> Path:  # we write summary files #
    out_dir = Path(out_dir)  # we normalize #
    out_dir.mkdir(parents=True, exist_ok=True)  # we ensure #
    runs = collect_runs(runs_dir)  # we collect #
    rows = [r.__dict__ for r in runs]  # we convert #
    with open(out_dir / "summary.json", "w") as f:  # we save #
        json.dump(rows, f, indent=2)  # we save #

    if len(runs) == 0:  # we guard #
        return out_dir / "summary.json"  # we return #

    labels = [Path(r.run_dir).name for r in runs]  # we set labels #
    errs = np.array([r.final_test_rel_l2 for r in runs], dtype=np.float64)  # we collect #
    order = np.argsort(errs)  # we sort #
    labels = [labels[i] for i in order]  # we reorder #
    errs = errs[order]  # we reorder #

    fig = plt.figure(figsize=(max(8, 0.35 * len(labels)), 4))  # we size #
    plt.bar(np.arange(len(labels)), errs)  # we plot #
    plt.yscale("log")  # we log scale #
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)  # we label #
    plt.ylabel("final test relative l2")  # we label #
    plt.title("Helmholtz operator benchmark (lower is better)")  # we title #
    plt.tight_layout()  # we layout #
    plt.savefig(out_dir / "summary_test_rel_l2.png", dpi=150)  # we save #
    plt.close(fig)  # we close #
    return out_dir / "summary.json"  # we return #


if __name__ == "__main__":  # we run cli #
    base = Path("experiments/helmholtz")  # we set base #
    write_summary(base / "runs", base / "runs" / "_summary")  # we write #


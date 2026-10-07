#!/usr/bin/env python3
"""Build the strict five-root meta-analysis after provenance repair finishes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


LIVE = Path("/root/MMNN/ICL-PDE")
PINNED = Path("/root/MMNN/.campaign-worktrees/icl-pde-225589f")
DATA = LIVE / "data/transformers"
REPAIR = DATA / "finite_basis_pde_coherent_repair_225589f_v1"
READY = REPAIR / "COHERENT_REPAIR_COMPLETE"
OUTPUT = DATA / "finite_basis_pde_coherent_meta_225589f_v1"
COMPLETE = OUTPUT / "COHERENT_META_COMPLETE"
PYTHON = Path("/root/MMNN/.venv/bin/python")
SUMMARIZER = PINNED / "experiments/transformers/summarize_finite_basis_pde_campaigns.py"

ROOTS = (
    DATA / "finite_basis_pde_minimal_v1",
    DATA / "finite_basis_pde_minimal_v1_spread",
    DATA / "finite_basis_pde_cold_controls_v1",
    DATA / "finite_basis_pde_loss_controls_v1",
    REPAIR,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    while not READY.exists():
        print(f"WAITING FOR {READY}", flush=True)
        time.sleep(60)
    command = [str(PYTHON), str(SUMMARIZER)]
    for root in ROOTS:
        command.extend(["--root", f"coherent225={root}"])
    command.extend(["--output", str(OUTPUT)])
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/mpl-finite-pde-coherent-meta")
    subprocess.run(command, cwd=PINNED, env=environment, check=True)

    required = (
        "meta_analysis.json",
        "REPORT.md",
        "checkpoint_metrics.csv",
        "matched_differences.csv",
        "matched_summary.csv",
        "checkpoint_overview.png",
        "checkpoint_overview.pdf",
        "matched_comparisons.png",
        "matched_comparisons.pdf",
    )
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"meta-analysis outputs missing: {missing}")
    payload = {
        "status": "complete",
        "created_unix": time.time(),
        "summarizer": str(SUMMARIZER),
        "summarizer_sha256": sha256(SUMMARIZER),
        "roots": [str(root) for root in ROOTS],
        "root_label": "coherent225",
        "artifacts": {name: sha256(OUTPUT / name) for name in required},
    }
    temporary = COMPLETE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(COMPLETE)
    print(f"COHERENT META COMPLETE {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create and semantically certify the 225589f-only publication meta-analysis."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


PINNED = Path("/root/MMNN/.campaign-worktrees/icl-pde-225589f")
LIVE = Path("/root/MMNN/ICL-PDE")
DATA = LIVE / "data/transformers"
PRIMARY = DATA / "finite_basis_pde_coherent_primary_view_225589f_v1"
AUXILIARY = DATA / "finite_basis_pde_coherent_auxiliary_view_225589f_v1"
OUTPUT = DATA / "finite_basis_pde_canonical_meta_225589f_v1"
PRIMARY_READY = PRIMARY / "CANONICAL_PRIMARY_COMPLETE"
AUXILIARY_READY = AUXILIARY / "CANONICAL_AUXILIARY_COMPLETE"
COMPLETE = OUTPUT / "CANONICAL_META_COMPLETE"
PYTHON = Path("/root/MMNN/.venv/bin/python")
SUMMARIZER = PINNED / "experiments/transformers/summarize_finite_basis_pde_campaigns.py"
EXPECTED_REVISION = "225589f62c0d10823755fa9f0ec68f605ef7f200"
EXPECTED_SOURCE_HASHES = {
    "exact_loop_transformer_decoder.py": "3d2cd9664159a97b13eaefdf7378746c974b7d86c26f05c0f55b6eb5e64c2441",
    "finite_basis_pde_icl_campaign.py": "a1bbb6ff2f0ec9f250c7a74527e421fadc50710051dce017400c917ba56eaa2b",
    "launch_finite_basis_pde_controls.py": "cacb1a4f558c5f1577db49386793bfd5d41fb8af06383f6041c60fa897b8f291",
    "launch_finite_basis_pde_spectral_controls.py": "ae808ea731fe50785b57818c2444a2e832516333890abf2b6bf7e842ca71a9a5",
    "launch_finite_basis_pde_sweep.py": "b1455357e6e1abee025032776095c4f86f65f5195d1ee59fe9ded762f9acdb03",
    "pure_icl_parametric_operator_richardson_attention.py": "4024ae1422eb77c1b1292ce0d2370be08e73ce93410a607fb092afc7308e2df8",
    "structured_one_head_heavyball.py": "026e497b4f19581e3207d9f3400ffaa65cd040b437e7f1b33de5a580233875e6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_all() -> None:
    while True:
        missing = [path for path in (PRIMARY_READY, AUXILIARY_READY) if not path.exists()]
        if not missing:
            return
        print(f"WAITING FOR {missing[0]} ({len(missing)} absent)", flush=True)
        time.sleep(60)


def validate_payload(payload: dict) -> dict:
    checkpoints = payload.get("checkpoints", [])
    differences = payload.get("matched_differences", [])
    issues = payload.get("issues", [])
    if len(checkpoints) != 62:
        raise RuntimeError(f"expected 62 canonical checkpoints, got {len(checkpoints)}")
    run_dirs = [str(row.get("run_dir")) for row in checkpoints]
    if len(run_dirs) != len(set(run_dirs)):
        raise RuntimeError("canonical checkpoint rows are not unique")
    for row in checkpoints:
        nuisance = json.loads(str(row["nuisance_config_json"]))
        source = nuisance["source_revision"]
        if source.get("git_revision_at_start") != EXPECTED_REVISION:
            raise RuntimeError(f"noncanonical revision in {row['run_dir']}")
        if source.get("campaign_source_sha256") != EXPECTED_SOURCE_HASHES[
            "finite_basis_pde_icl_campaign.py"
        ]:
            raise RuntimeError(f"noncanonical campaign hash in {row['run_dir']}")
        if source.get("source_sha256") != EXPECTED_SOURCE_HASHES:
            raise RuntimeError(f"noncanonical source map in {row['run_dir']}")
        if row.get("solution_nmse_source") != "functional_generalization_audit":
            raise RuntimeError(f"noncanonical solution estimand in {row['run_dir']}")
    if not differences:
        raise RuntimeError("no paired treatment differences were produced")
    if issues:
        raise RuntimeError(f"canonical summarizer reported {len(issues)} issues")
    return {
        "checkpoint_rows": len(checkpoints),
        "paired_metric_differences": len(differences),
        "issues": 0,
        "all_revision_225589f": True,
        "all_source_hashes_verified": True,
        "all_solution_metrics_functional_schema": True,
    }


def main() -> None:
    wait_for_all()
    command = [
        str(PYTHON),
        str(SUMMARIZER),
        "--root",
        f"coherent225={PRIMARY}",
        "--root",
        f"coherent225={AUXILIARY}",
        "--output",
        str(OUTPUT),
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/mpl-finite-pde-canonical-meta")
    subprocess.run(command, cwd=PINNED, env=environment, check=True)
    meta_path = OUTPUT / "meta_analysis.json"
    validation = validate_payload(json.loads(meta_path.read_text()))
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
        raise RuntimeError(f"canonical outputs missing: {missing}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "created_unix": time.time(),
        "validation": validation,
        "root_label": "coherent225",
        "roots": [str(PRIMARY), str(AUXILIARY)],
        "summarizer": str(SUMMARIZER),
        "summarizer_sha256": sha256(SUMMARIZER),
        "artifacts": {name: sha256(OUTPUT / name) for name in required},
    }
    temporary = COMPLETE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(COMPLETE)
    print(f"CANONICAL META COMPLETE {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

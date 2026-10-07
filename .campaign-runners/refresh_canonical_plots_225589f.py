#!/usr/bin/env python3
"""Regenerate every canonical per-run figure with the pinned 225589f plotter."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


PINNED = Path("/root/MMNN/.campaign-worktrees/icl-pde-225589f")
LIVE = Path("/root/MMNN/ICL-PDE")
DATA = LIVE / "data/transformers"
PRIMARY = DATA / "finite_basis_pde_coherent_primary_view_225589f_v1"
AUXILIARY = DATA / "finite_basis_pde_coherent_auxiliary_view_225589f_v1"
CERTIFICATE_ROOT = DATA / "finite_basis_pde_publication_certificate_225589f_v1"
READY = CERTIFICATE_ROOT / "CANONICAL_PLOTS_REFRESHED"
PYTHON = Path("/root/MMNN/.venv/bin/python")
PLOTTER = PINNED / "experiments/transformers/plot_finite_basis_pde_campaign.py"
EXPECTED_REVISION = "225589f62c0d10823755fa9f0ec68f605ef7f200"
EXPECTED_PLOTTER_SHA256 = (
    "bf592225bf9de794d95c090fe92f5d18910649cde8b64508257c751023c564dc"
)
EXPECTED_EXECUTABLE_HASHES = {
    "exact_loop_transformer_decoder.py": "3d2cd9664159a97b13eaefdf7378746c974b7d86c26f05c0f55b6eb5e64c2441",
    "finite_basis_pde_icl_campaign.py": "a1bbb6ff2f0ec9f250c7a74527e421fadc50710051dce017400c917ba56eaa2b",
    "launch_finite_basis_pde_controls.py": "cacb1a4f558c5f1577db49386793bfd5d41fb8af06383f6041c60fa897b8f291",
    "launch_finite_basis_pde_spectral_controls.py": "ae808ea731fe50785b57818c2444a2e832516333890abf2b6bf7e842ca71a9a5",
    "launch_finite_basis_pde_sweep.py": "b1455357e6e1abee025032776095c4f86f65f5195d1ee59fe9ded762f9acdb03",
    "pure_icl_parametric_operator_richardson_attention.py": "4024ae1422eb77c1b1292ce0d2370be08e73ce93410a607fb092afc7308e2df8",
    "structured_one_head_heavyball.py": "026e497b4f19581e3207d9f3400ffaa65cd040b437e7f1b33de5a580233875e6",
}
FIGURE_STEMS = (
    "reconstruction_examples",
    "error_decomposition",
    "decoder_depth",
    "training_and_encoder",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def wait_for_all(paths: tuple[Path, ...]) -> None:
    while True:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            return
        print(f"WAITING FOR {missing[0]} ({len(missing)} absent)", flush=True)
        time.sleep(60)


def verify_pinned_toolchain() -> dict[str, str]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PINNED, text=True
    ).strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"pinned worktree moved: {revision}")
    if sha256(PLOTTER) != EXPECTED_PLOTTER_SHA256:
        raise RuntimeError("pinned plotter bytes changed")
    transformer_dir = PINNED / "experiments/transformers"
    observed = {
        name: sha256(transformer_dir / name) for name in EXPECTED_EXECUTABLE_HASHES
    }
    if observed != EXPECTED_EXECUTABLE_HASHES:
        raise RuntimeError("pinned executable source bytes changed")
    return observed


def view_runs(root: Path, expected_count: int) -> list[Path]:
    manifest_path = root / "CANONICAL_VIEW_MANIFEST.json"
    payload = json.loads(manifest_path.read_text())
    entries = payload.get("runs", [])
    if payload.get("expected_runs") != expected_count or len(entries) != expected_count:
        raise RuntimeError(f"wrong canonical manifest count in {manifest_path}")
    runs = [root / str(entry["name"]) for entry in entries]
    if len(set(runs)) != expected_count:
        raise RuntimeError(f"duplicate canonical run names in {manifest_path}")
    expected_complete = {run.resolve() for run in runs}
    observed_complete = {
        marker.parent.resolve() for marker in root.rglob("COMPLETE")
    }
    if observed_complete != expected_complete:
        raise RuntimeError(
            f"canonical COMPLETE set mismatch in {root}: "
            f"missing={sorted(map(str, expected_complete - observed_complete))}, "
            f"extra={sorted(map(str, observed_complete - expected_complete))}"
        )
    return runs


def verify_render(path: Path, started_ns: int) -> None:
    if not path.is_file() or path.stat().st_size < 64:
        raise RuntimeError(f"missing or empty freshly rendered artifact: {path}")
    if path.stat().st_mtime_ns < started_ns:
        raise RuntimeError(f"artifact was not refreshed by pinned plotter: {path}")
    prefix = path.read_bytes()[:8]
    if path.suffix == ".png" and prefix != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"invalid PNG signature: {path}")
    if path.suffix == ".pdf" and not prefix.startswith(b"%PDF-"):
        raise RuntimeError(f"invalid PDF signature: {path}")


def refresh_run(run_dir: Path, environment: dict[str, str]) -> dict[str, str]:
    verify_pinned_toolchain()
    started_ns = time.time_ns()
    subprocess.run(
        [
            str(PYTHON),
            str(PLOTTER),
            "--run",
            str(run_dir),
            "--device",
            "cpu",
        ],
        cwd=PINNED,
        env=environment,
        check=True,
    )
    artifacts = [
        run_dir / f"{stem}.{suffix}"
        for stem in FIGURE_STEMS
        for suffix in ("png", "pdf")
    ]
    artifacts.append(run_dir / "reconstruction_examples.pt")
    for path in artifacts:
        if path.suffix in {".png", ".pdf"}:
            verify_render(path, started_ns)
        elif not path.is_file() or path.stat().st_size < 64:
            raise RuntimeError(f"missing reconstruction payload: {path}")
    verify_pinned_toolchain()
    return {path.name: sha256(path) for path in artifacts}


def verify_aggregate(root: Path, expected_rows: int) -> dict[str, str]:
    verify_pinned_toolchain()
    artifacts = (
        root / "aggregate_metrics.csv",
        root / "task_diversity_transition.png",
        root / "task_diversity_transition.pdf",
    )
    for path in artifacts:
        if not path.is_file() or path.stat().st_size < 64:
            raise RuntimeError(f"missing canonical aggregate artifact: {path}")
        if path.suffix in {".png", ".pdf"}:
            prefix = path.read_bytes()[:8]
            if path.suffix == ".png" and prefix != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError(f"invalid aggregate PNG signature: {path}")
            if path.suffix == ".pdf" and not prefix.startswith(b"%PDF-"):
                raise RuntimeError(f"invalid aggregate PDF signature: {path}")
    with (root / "aggregate_metrics.csv").open(newline="") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    if row_count != expected_rows:
        raise RuntimeError(
            f"canonical aggregate row count mismatch in {root}: "
            f"{row_count} != {expected_rows}"
        )
    verify_pinned_toolchain()
    return {path.name: sha256(path) for path in artifacts}


def main() -> None:
    if READY.exists():
        raise RuntimeError(f"one-shot plot certificate already exists: {READY}")
    wait_for_all(
        (
            PRIMARY / "CANONICAL_PRIMARY_COMPLETE",
            AUXILIARY / "CANONICAL_AUXILIARY_COMPLETE",
        )
    )
    verify_pinned_toolchain()
    runs = [*view_runs(PRIMARY, 44), *view_runs(AUXILIARY, 18)]
    if len(runs) != 62 or len(set(runs)) != 62:
        raise RuntimeError("expected 62 unique canonical run directories")
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/tmp/mpl-finite-pde-canonical-refresh"
    per_run: dict[str, dict[str, str]] = {}
    for index, run_dir in enumerate(runs, start=1):
        print(f"REFRESHING {index}/62 {run_dir.name}", flush=True)
        per_run[str(run_dir.resolve())] = refresh_run(run_dir, environment)
    aggregates = {
        str(PRIMARY.resolve()): verify_aggregate(PRIMARY, 44),
        str(AUXILIARY.resolve()): verify_aggregate(AUXILIARY, 6),
    }
    toolchain = verify_pinned_toolchain()
    atomic_json(
        READY,
        {
            "status": "complete",
            "created_unix": time.time(),
            "pinned_revision": EXPECTED_REVISION,
            "plotter": str(PLOTTER),
            "plotter_sha256": EXPECTED_PLOTTER_SHA256,
            "executable_source_sha256": toolchain,
            "run_count": len(runs),
            "per_run_artifact_sha256": per_run,
            "aggregate_artifact_sha256": aggregates,
        },
    )
    print(f"CANONICAL PLOTS REFRESHED {READY}", flush=True)


if __name__ == "__main__":
    main()

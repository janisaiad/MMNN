#!/usr/bin/env python3
"""Run the minimal provenance repair from the immutable 225589f worktree.

This file is intentionally outside the live repository.  It never overwrites an
archived result: repaired runs go to a separate root and retain their original
scientific arguments.  Four persistent workers may execute disjoint shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


PINNED_ROOT = Path("/root/MMNN/.campaign-worktrees/icl-pde-225589f")
LIVE_ROOT = Path("/root/MMNN/ICL-PDE")
DATA_ROOT = LIVE_ROOT / "data/transformers"
OUTPUT_ROOT = DATA_ROOT / "finite_basis_pde_coherent_repair_225589f_v1"
MANIFEST_PATH = OUTPUT_ROOT / "COHERENT_REPAIR_MANIFEST.json"
RUNNER_SNAPSHOT = OUTPUT_ROOT / "RUNNER_SNAPSHOT.py"
BARRIER_PATH = DATA_ROOT / "FINITE_BASIS_PDE_EXISTING_QUEUES_COMPLETE"
PYTHON = Path("/root/MMNN/.venv/bin/python")
CAMPAIGN = PINNED_ROOT / "experiments/transformers/finite_basis_pde_icl_campaign.py"

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

PREREQUISITES = (
    DATA_ROOT / "finite_basis_pde_minimal_v1/CAMPAIGN_COMPLETE",
    DATA_ROOT / "finite_basis_pde_minimal_v1_spread/CAMPAIGN_COMPLETE",
    DATA_ROOT / "finite_basis_pde_cold_controls_v1/CAMPAIGN_COMPLETE",
    DATA_ROOT / "finite_basis_pde_loss_controls_v1/CAMPAIGN_COMPLETE",
    DATA_ROOT / "finite_basis_pde_objective_controls_v1/CAMPAIGN_COMPLETE",
    DATA_ROOT / "finite_basis_pde_leading_5k_control_v1/CAMPAIGN_COMPLETE",
    DATA_ROOT / "finite_basis_pde_rank4_spread_v1/CAMPAIGN_COMPLETE",
    DATA_ROOT / "finite_basis_pde_rank4_spread_cold_v1/CAMPAIGN_COMPLETE",
    DATA_ROOT
    / "finite_basis_pde_spectral_controls_v1/SPECTRAL_CAMPAIGN_COMPLETE",
)

# (phase, source manifest below the pinned repository, original run name)
SELECTIONS = (
    # Primary cells needed for strict K/replica claims (14 runs).
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k4.json", "k4_m0_2_query_attention_seed0"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k4.json", "k4_m0_2_query_attention_seed1"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k4.json", "k4_m0_2_query_attention_seed2"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k4.json", "k4_m0_4_query_attention_seed0"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8.json", "k8_m0_4_query_attention_seed0"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8.json", "k8_m0_4_query_attention_seed1"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8.json", "k8_m0_4_query_attention_seed2"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8_observable.json", "k8_m0_4_query_attention_r16_m64_seed0"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8_observable.json", "k8_m0_4_query_attention_r16_m64_seed1"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8_observable.json", "k8_m0_8_query_attention_r16_m64_seed0"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8_observable.json", "k8_m0_8_query_attention_r16_m64_seed1"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1_spread/MANIFEST_k8_spread.json", "k8_m0_4_query_attention_spread_seed0"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1_spread/MANIFEST_k8_spread.json", "k8_m0_4_query_attention_spread_seed1"),
    ("primary", "data/transformers/finite_basis_pde_minimal_v1_spread/MANIFEST_k8_spread.json", "k8_m0_4_query_attention_spread_seed2"),
    # Auxiliary cold/objective controls (6 runs), started only after all 14 above.
    ("auxiliary", "data/transformers/finite_basis_pde_cold_controls_v1/CONTROL_MANIFEST.json", "k4_m0_2_query_cold_seed0"),
    ("auxiliary", "data/transformers/finite_basis_pde_cold_controls_v1/CONTROL_MANIFEST.json", "k4_m0_2_query_cold_seed1"),
    ("auxiliary", "data/transformers/finite_basis_pde_cold_controls_v1/CONTROL_MANIFEST.json", "k4_m0_2_query_cold_seed2"),
    ("auxiliary", "data/transformers/finite_basis_pde_loss_controls_v1/CONTROL_MANIFEST.json", "k4_m0_8_query_prompt_seed1"),
    ("auxiliary", "data/transformers/finite_basis_pde_loss_controls_v1/CONTROL_MANIFEST.json", "k4_m0_8_query_prompt_seed2"),
    ("auxiliary", "data/transformers/finite_basis_pde_loss_controls_v1/CONTROL_MANIFEST.json", "k4_m0_8_query_operator_seed0"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_pinned_source() -> None:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PINNED_ROOT, text=True
    ).strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"pinned worktree moved: {revision} != {EXPECTED_REVISION}")
    source_dir = PINNED_ROOT / "experiments/transformers"
    actual = {name: sha256(source_dir / name) for name in EXPECTED_SOURCE_HASHES}
    if actual != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"pinned executable bytes changed: {actual}")


def selected_entries() -> list[dict[str, Any]]:
    cache: dict[str, dict[str, dict[str, Any]]] = {}
    entries: list[dict[str, Any]] = []
    for phase, relative_manifest, name in SELECTIONS:
        if relative_manifest not in cache:
            manifest_path = PINNED_ROOT / relative_manifest
            payload = json.loads(manifest_path.read_text())
            indexed = {str(entry["name"]): entry for entry in payload["runs"]}
            if len(indexed) != len(payload["runs"]):
                raise RuntimeError(f"duplicate names in {manifest_path}")
            cache[relative_manifest] = indexed
        try:
            source_entry = cache[relative_manifest][name]
        except KeyError as error:
            raise RuntimeError(f"{name} absent from {relative_manifest}") from error
        arguments = [str(value) for value in source_entry["arguments"]]
        flags = [value for value in arguments if value.startswith("--")]
        if len(flags) != len(set(flags)):
            raise RuntimeError(f"duplicate CLI flag in selected run {name}")
        if "--outdir" in flags or "--resume" in flags:
            raise RuntimeError(f"bookkeeping flag embedded in selected run {name}")
        entries.append(
            {
                "name": name,
                "phase": phase,
                "source_manifest": relative_manifest,
                "source_manifest_sha256": sha256(PINNED_ROOT / relative_manifest),
                "arguments": arguments,
            }
        )
    names = [str(entry["name"]) for entry in entries]
    if len(names) != len(set(names)):
        raise RuntimeError("repair selections contain duplicate output names")
    return entries


def prepare_manifest() -> list[dict[str, Any]]:
    verify_pinned_source()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    entries = selected_entries()
    runner_bytes = Path(__file__).read_bytes()
    payload = {
        "schema_version": 2,
        "purpose": "minimal strict-provenance repair; archived runs are never overwritten",
        "expected_runs": len(entries),
        "phase_counts": {
            phase: sum(entry["phase"] == phase for entry in entries)
            for phase in ("primary", "auxiliary")
        },
        "pinned_git_revision": EXPECTED_REVISION,
        "pinned_source_sha256": EXPECTED_SOURCE_HASHES,
        "runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
        "barrier": str(BARRIER_PATH),
        "prerequisites": [str(path) for path in PREREQUISITES],
        "runs": entries,
    }
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text())
        if existing != payload:
            raise RuntimeError(f"existing repair manifest drift: {MANIFEST_PATH}")
    else:
        atomic_json(MANIFEST_PATH, payload)
    if RUNNER_SNAPSHOT.exists():
        if RUNNER_SNAPSHOT.read_bytes() != runner_bytes:
            raise RuntimeError(f"runner snapshot drift: {RUNNER_SNAPSHOT}")
    else:
        RUNNER_SNAPSHOT.write_bytes(runner_bytes)
    return entries


def wait_for_paths(paths: tuple[Path, ...] | list[Path], label: str) -> None:
    while True:
        missing = [path for path in paths if not path.exists()]
        if not missing:
            return
        print(
            f"WAITING {label}: {len(missing)}/{len(paths)} markers absent; "
            f"next={missing[0]}",
            flush=True,
        )
        time.sleep(60)


def build_barrier() -> None:
    prepare_manifest()
    wait_for_paths(list(PREREQUISITES), "existing queues")
    payload = {
        "status": "complete",
        "created_unix": time.time(),
        "pinned_git_revision": EXPECTED_REVISION,
        "prerequisites": {str(path): sha256(path) for path in PREREQUISITES},
    }
    atomic_json(BARRIER_PATH, payload)
    print(f"BARRIER READY {BARRIER_PATH}", flush=True)


def verify_completed_run(run_dir: Path, arguments: list[str]) -> None:
    run_path = run_dir / "run.json"
    complete_path = run_dir / "COMPLETE"
    if not run_path.is_file() or not complete_path.is_file():
        raise RuntimeError(f"run did not complete atomically: {run_dir}")
    run = json.loads(run_path.read_text())
    if run.get("git_revision_at_start") != EXPECTED_REVISION:
        raise RuntimeError(f"wrong git revision in {run_path}")
    source = run.get("source_sha256_at_initial_start", run.get("source_sha256"))
    if source != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"wrong source hashes in {run_path}")
    if str(run.get("status")) != "complete":
        raise RuntimeError(f"COMPLETE marker with noncomplete run record: {run_path}")
    expected_args = dict(zip(arguments[0::2], arguments[1::2], strict=True))
    saved_args = run.get("args", {})
    for flag, expected in expected_args.items():
        key = flag.removeprefix("--").replace("-", "_")
        saved = saved_args.get(key)
        equal = str(saved).lower() == expected.lower()
        if not equal and isinstance(saved, (int, float)) and not isinstance(saved, bool):
            try:
                equal = float(saved) == float(expected)
            except ValueError:
                equal = False
        if not equal:
            raise RuntimeError(
                f"argument drift in {run_path}: {key}={saved!r} != {expected!r}"
            )


def run_one(entry: dict[str, Any], worker: int) -> None:
    verify_pinned_source()
    run_dir = OUTPUT_ROOT / str(entry["name"])
    arguments = [str(value) for value in entry["arguments"]]
    if (run_dir / "COMPLETE").is_file():
        verify_completed_run(run_dir, arguments)
        print(f"WORKER {worker} SKIP COMPLETE {run_dir.name}", flush=True)
        return
    command = [str(PYTHON), str(CAMPAIGN), "--outdir", str(run_dir), *arguments]
    if (run_dir / "run.json").exists():
        latest = run_dir / "model_latest.pt"
        if not latest.is_file():
            raise RuntimeError(f"partial run has no resumable checkpoint: {run_dir}")
        command.extend(["--resume", str(latest)])
    print(f"WORKER {worker} START {run_dir.name}", flush=True)
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/mpl-finite-pde-repair")
    subprocess.run(command, cwd=PINNED_ROOT, env=environment, check=True)
    verify_completed_run(run_dir, arguments)
    print(f"WORKER {worker} COMPLETE {run_dir.name}", flush=True)


def run_worker(worker: int, workers: int) -> None:
    if workers <= 0 or not 0 <= worker < workers:
        raise ValueError("worker must lie in [0, workers)")
    entries = prepare_manifest()
    wait_for_paths([BARRIER_PATH], "repair barrier")
    by_phase = {
        phase: [entry for entry in entries if entry["phase"] == phase]
        for phase in ("primary", "auxiliary")
    }
    for phase in ("primary", "auxiliary"):
        if phase == "auxiliary":
            primary_markers = tuple(
                OUTPUT_ROOT / str(entry["name"]) / "COMPLETE"
                for entry in by_phase["primary"]
            )
            wait_for_paths(list(primary_markers), "primary provenance repair")
        selected = [
            entry
            for index, entry in enumerate(by_phase[phase])
            if index % workers == worker
        ]
        print(
            f"WORKER {worker}/{workers} PHASE {phase}: {len(selected)} runs",
            flush=True,
        )
        for entry in selected:
            run_one(entry, worker)
    print(f"WORKER {worker}/{workers} DONE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("manifest-only", "barrier", "worker"), required=True
    )
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "manifest-only":
        entries = prepare_manifest()
        print(f"MANIFEST READY {MANIFEST_PATH} ({len(entries)} runs)")
    elif args.mode == "barrier":
        build_barrier()
    else:
        run_worker(args.worker, args.workers)


if __name__ == "__main__":
    main()

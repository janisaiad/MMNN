#!/usr/bin/env python3
"""Snapshot publication-safe 225589f cohorts after all repair runs finish."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


LIVE = Path("/root/MMNN/ICL-PDE")
PINNED = Path("/root/MMNN/.campaign-worktrees/icl-pde-225589f")
DATA = LIVE / "data/transformers"
REPAIR = DATA / "finite_basis_pde_coherent_repair_225589f_v1"
REPAIR_READY = REPAIR / "COHERENT_REPAIR_COMPLETE"
PRIMARY_VIEW = DATA / "finite_basis_pde_coherent_primary_view_225589f_v1"
AUXILIARY_VIEW = DATA / "finite_basis_pde_coherent_auxiliary_view_225589f_v1"
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

PRIMARY_SOURCES = (
    (
        "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k4.json",
        DATA / "finite_basis_pde_minimal_v1",
    ),
    (
        "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8.json",
        DATA / "finite_basis_pde_minimal_v1",
    ),
    (
        "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8_observable.json",
        DATA / "finite_basis_pde_minimal_v1",
    ),
    (
        "data/transformers/finite_basis_pde_minimal_v1_spread/MANIFEST_k8_spread.json",
        DATA / "finite_basis_pde_minimal_v1_spread",
    ),
)

AUXILIARY_SOURCES = (
    (
        "data/transformers/finite_basis_pde_cold_controls_v1/CONTROL_MANIFEST.json",
        DATA / "finite_basis_pde_cold_controls_v1",
    ),
    (
        "data/transformers/finite_basis_pde_loss_controls_v1/CONTROL_MANIFEST.json",
        DATA / "finite_basis_pde_loss_controls_v1",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def wait_for(path: Path) -> None:
    while not path.exists():
        print(f"WAITING FOR {path}", flush=True)
        time.sleep(60)


def argument_map(arguments: list[Any]) -> dict[str, str]:
    values = [str(item) for item in arguments]
    if len(values) % 2:
        raise RuntimeError(f"argument list is not flag/value paired: {values}")
    result: dict[str, str] = {}
    for flag, value in zip(values[0::2], values[1::2], strict=True):
        if not flag.startswith("--"):
            raise RuntimeError(f"expected CLI flag, got {flag!r}")
        result[flag] = value
    return result


def is_primary(entry: dict[str, Any]) -> bool:
    args = argument_map(entry["arguments"])
    return (
        args.get("--loss-profile") == "query_only"
        and args.get("--init") == "true_noisy"
        and args.get("--solver") == "dual_attention_richardson"
        and args.get("--attention-mode") == "linear"
        and args.get("--qk-constraint", "untied") == "untied"
        and args.get("--loop-preconditioner-head", "coordinate_ritz")
        == "coordinate_ritz"
    )


def load_entries(relative_manifest: str) -> list[dict[str, Any]]:
    payload = json.loads((PINNED / relative_manifest).read_text())
    return [dict(entry) for entry in payload["runs"]]


def numeric_equal(saved: Any, expected: str) -> bool:
    if str(saved).lower() == expected.lower():
        return True
    if isinstance(saved, (int, float)) and not isinstance(saved, bool):
        try:
            return float(saved) == float(expected)
        except ValueError:
            return False
    return False


def verify_source_run(source: Path, entry: dict[str, Any]) -> None:
    required = (
        "COMPLETE",
        "run.json",
        "summary.json",
        "model_final.pt",
        "task_bank.pt",
        "train.csv",
        "evaluation.csv",
        "decoder_depth.csv",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"missing source artifacts in {source}: {missing}")
    run = json.loads((source / "run.json").read_text())
    if run.get("status") != "complete":
        raise RuntimeError(f"noncomplete source run: {source}")
    if run.get("git_revision_at_start") != EXPECTED_REVISION:
        raise RuntimeError(f"wrong source revision: {source}")
    hashes = run.get("source_sha256_at_initial_start", run.get("source_sha256"))
    if hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"wrong executable hashes: {source}")
    saved_args = run.get("args", {})
    for flag, expected in argument_map(entry["arguments"]).items():
        key = flag.removeprefix("--").replace("-", "_")
        if not numeric_equal(saved_args.get(key), expected):
            raise RuntimeError(
                f"argument mismatch in {source}: {key}={saved_args.get(key)!r} "
                f"!= {expected!r}"
            )


def snapshot_run(source: Path, destination: Path) -> None:
    if destination.exists():
        if not (destination / "COMPLETE").is_file():
            raise RuntimeError(f"partial canonical snapshot already exists: {destination}")
        if sha256(destination / "model_final.pt") != sha256(source / "model_final.pt"):
            raise RuntimeError(f"canonical snapshot model drift: {destination}")
        if sha256(destination / "run.json") != sha256(source / "run.json"):
            raise RuntimeError(f"canonical snapshot provenance drift: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copytree(source, temporary)
    temporary.replace(destination)


def repair_index() -> dict[str, tuple[str, dict[str, Any]]]:
    payload = json.loads((REPAIR / "COHERENT_REPAIR_MANIFEST.json").read_text())
    index = {
        str(entry["name"]): (str(entry["phase"]), dict(entry))
        for entry in payload["runs"]
    }
    if len(index) != 20:
        raise RuntimeError("repair manifest must contain exactly 20 unique runs")
    return index


def build_primary(repairs: dict[str, tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for relative_manifest, original_root in PRIMARY_SOURCES:
        for entry in load_entries(relative_manifest):
            if not is_primary(entry):
                continue
            name = str(entry["name"])
            if name in repairs:
                phase, repair_entry = repairs[name]
                if phase != "primary" or repair_entry["arguments"] != entry["arguments"]:
                    raise RuntimeError(f"invalid primary repair entry: {name}")
                source = REPAIR / name
            else:
                source = original_root / name
            selected.append({**entry, "source_run_dir": str(source)})
    if len(selected) != 44 or len({entry["name"] for entry in selected}) != 44:
        raise RuntimeError(f"expected 44 unique primary runs, got {len(selected)}")
    if sum(entry["name"] in repairs for entry in selected) != 14:
        raise RuntimeError("primary view must contain exactly 14 repaired runs")
    return selected


def build_auxiliary(
    repairs: dict[str, tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for relative_manifest, original_root in AUXILIARY_SOURCES:
        for entry in load_entries(relative_manifest):
            name = str(entry["name"])
            if name in repairs:
                phase, repair_entry = repairs[name]
                if phase != "auxiliary" or repair_entry["arguments"] != entry["arguments"]:
                    raise RuntimeError(f"invalid auxiliary repair entry: {name}")
                source = REPAIR / name
            else:
                source = original_root / name
            selected.append({**entry, "source_run_dir": str(source)})

    # The loss-control manifest intentionally supplied only prompt seeds 1 and 2;
    # seed 0 lives in the original K4 manifest.
    k4_entries = {
        str(entry["name"]): entry
        for entry in load_entries(
            "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k4.json"
        )
    }
    prompt_seed0 = dict(k4_entries["k4_m0_8_query_prompt_attention_seed0"])
    selected.append(
        {
            **prompt_seed0,
            "source_run_dir": str(
                DATA
                / "finite_basis_pde_minimal_v1"
                / "k4_m0_8_query_prompt_attention_seed0"
            ),
        }
    )
    if len(selected) != 18 or len({entry["name"] for entry in selected}) != 18:
        raise RuntimeError(f"expected 18 unique auxiliary runs, got {len(selected)}")
    if sum(entry["name"] in repairs for entry in selected) != 6:
        raise RuntimeError("auxiliary view must contain exactly 6 repaired runs")
    return selected


def build_view(root: Path, phase: str, entries: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        source = Path(str(entry["source_run_dir"]))
        verify_source_run(source, entry)
        snapshot_run(source, root / str(entry["name"]))
    manifest = {
        "schema_version": 2,
        "purpose": f"publication-safe {phase} snapshot; only exact 225589f runs",
        "phase": phase,
        "expected_runs": len(entries),
        "pinned_git_revision": EXPECTED_REVISION,
        "pinned_source_sha256": EXPECTED_SOURCE_HASHES,
        "runs": entries,
    }
    manifest_path = root / "CANONICAL_VIEW_MANIFEST.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f"canonical manifest drift: {manifest_path}")
    atomic_json(manifest_path, manifest)
    atomic_json(
        root / "VIEW_READY",
        {
            "status": "ready",
            "created_unix": time.time(),
            "phase": phase,
            "runs": len(entries),
            "manifest_sha256": sha256(manifest_path),
        },
    )
    print(f"VIEW READY {phase} {root} ({len(entries)} runs)", flush=True)


def main() -> None:
    wait_for(REPAIR_READY)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PINNED, text=True
    ).strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"pinned worktree moved: {revision}")
    repairs = repair_index()
    build_view(PRIMARY_VIEW, "primary", build_primary(repairs))
    build_view(AUXILIARY_VIEW, "auxiliary", build_auxiliary(repairs))


if __name__ == "__main__":
    main()

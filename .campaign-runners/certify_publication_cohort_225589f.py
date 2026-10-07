#!/usr/bin/env python3
"""Independent final gate for the exact 62-run canonical PDE cohort."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-finite-pde-publication-gate")

import torch


PINNED = Path("/root/MMNN/.campaign-worktrees/icl-pde-225589f")
LIVE = Path("/root/MMNN/ICL-PDE")
DATA = LIVE / "data/transformers"
PRIMARY = DATA / "finite_basis_pde_coherent_primary_view_225589f_v1"
AUXILIARY = DATA / "finite_basis_pde_coherent_auxiliary_view_225589f_v1"
META = DATA / "finite_basis_pde_canonical_meta_225589f_v1"
CERTIFICATE_ROOT = DATA / "finite_basis_pde_publication_certificate_225589f_v1"
PLOTS_READY = CERTIFICATE_ROOT / "CANONICAL_PLOTS_REFRESHED"
META_READY = META / "CANONICAL_META_COMPLETE"
CERTIFIED = CERTIFICATE_ROOT / "PUBLICATION_COHORT_CERTIFIED"
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
EXPECTED_TOOL_HASHES = {
    "experiments/transformers/plot_finite_basis_pde_campaign.py": "bf592225bf9de794d95c090fe92f5d18910649cde8b64508257c751023c564dc",
    "experiments/transformers/summarize_finite_basis_pde_campaigns.py": "e2181ac99d426bcc0d6019c161809a18311f17cbe4dedfb3e2e729de0078165f",
    "experiments/transformers/finalize_finite_basis_pde_campaign.py": "e97dab6ce13ba1c5da2dce025d421700e038b07ce6e437cf1d1b94e19d007a4f",
    "experiments/transformers/audit_finite_basis_replica_bounds.py": "50c0040cdfdea81221a70a9e34662424a3fbcfc0e640c621e668d77a0205ace4",
    "experiments/transformers/audit_finite_basis_functional_generalization.py": "6efdc5b6dd2d99f0172fa20d403e488e109a62b77795a78ebb9906ae2e23d3ea",
    "experiments/transformers/audit_decoder_depth_precision.py": "07259849b67d23b9d816fe562f4f496c556e2b2138b3161713ac8a6371fb83e6",
    "experiments/transformers/audit_learned_stationary_decoder.py": "00a5b7f951084ea3fadfac92852efc75a726580148e339b9bec9f7308179d640",
    "experiments/transformers/audit_prompt_numerical_precision.py": "b6b0c249ce5f4bd8af5716529f2c7b67baae93c9850656a9759fe892c79c94bb",
    "experiments/transformers/first_principles_decoder_cells.py": "9d8cba0f7cf7d75c320ee9115dcb06ac0c655853f51ebe8a19c1e62e2118b592",
    "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k4.json": "0a930b99ea9a14fd54390070f6e860c2005552e89ab4876bc3a30926740535d1",
    "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8.json": "41b8e7d27d54e5fb1a081066d5a2918af0a9aabe7a3d34249a26aa67f2c28d4f",
    "data/transformers/finite_basis_pde_minimal_v1/MANIFEST_k8_observable.json": "3fb0bfe1ba788e2cecf653321eb305f0bb7cbbff420087176fe207b1ab69f2c6",
    "data/transformers/finite_basis_pde_minimal_v1_spread/MANIFEST_k8_spread.json": "220cd1368c49af1765f7e097a0b1955bf1c5ca2b5c50d4b1851b48c74e0684d8",
    "data/transformers/finite_basis_pde_cold_controls_v1/CONTROL_MANIFEST.json": "427728ae8e5f887cbf0d3869d123dfc735da225242b423b76add4951af62e596",
    "data/transformers/finite_basis_pde_loss_controls_v1/CONTROL_MANIFEST.json": "8f16c70e5f59e4dbec2e2bb507001143a67064aee478273bab22f2692faf7be5",
}
EXPECTED_EXTERNAL_HASHES = {
    Path("/root/MMNN/.campaign-runners/build_canonical_views_225589f.py"): "a16bbe5aaa29b14d4d427770539a3df32b8a7628113d0da4ab573eb0201f0aa3",
    Path("/root/MMNN/.campaign-runners/finalize_canonical_meta_225589f.py"): "52656ef349d765055ad4ed56d27c069bf044d7eaae93e8a7f4dbf0ace117f721",
    DATA / "finite_basis_pde_coherent_repair_225589f_v1/COHERENT_REPAIR_MANIFEST.json": "b698bde9ecd1442fcfc29a0dd3f612a8eb3507ee42d17f09a36f74ac505e61c1",
}
ARGUMENT_EXCLUSIONS = {"outdir", "resume", "allow_completed"}
METRIC_FIELDS = {
    "solution_nmse",
    "coefficient_full_nmse",
    "coefficient_perturbation_nmse",
    "physical_z_nmse",
    "active_subspace_error",
    "full_subspace_error",
}
EXPECTED_METRIC_PROVENANCE = {
    "solution_nmse": ("functional_generalization_audit", "2", "hash_verified"),
    "coefficient_full_nmse": (
        "functional_generalization_audit",
        "2",
        "hash_verified",
    ),
    "coefficient_perturbation_nmse": (
        "functional_generalization_audit",
        "2",
        "hash_verified",
    ),
    "physical_z_nmse": (
        "functional_generalization_audit",
        "2",
        "hash_verified",
    ),
    "active_subspace_error": (
        "replica_observability_audit",
        "2",
        "hash_verified",
    ),
    "full_subspace_error": (
        "native_summary_checkpoint",
        "native-v1",
        "checkpoint_hash_bound",
    ),
}
RAW_ARTIFACTS = (
    "COMPLETE",
    "run.json",
    "summary.json",
    "model_final.pt",
    "task_bank.pt",
    "train.csv",
    "evaluation.csv",
    "decoder_depth.csv",
    "bound_and_observability_audit.json",
)
PLOT_ARTIFACTS = tuple(
    f"{stem}.{suffix}"
    for stem in (
        "reconstruction_examples",
        "error_decomposition",
        "decoder_depth",
        "training_and_encoder",
    )
    for suffix in ("png", "pdf")
)
FRESH_PLOT_ARTIFACTS = {*PLOT_ARTIFACTS, "reconstruction_examples.pt"}
AGGREGATE_ARTIFACTS = {
    "aggregate_metrics.csv",
    "task_diversity_transition.png",
    "task_diversity_transition.pdf",
}
MATCHED_SUMMARY_FIELDS = (
    "treatment_a",
    "treatment_b",
    "K",
    "active_rank",
    "M0",
    "R",
    "m",
    "depth",
    "d",
    "steps",
    "nuisance_config_id",
    "metric",
    "source",
    "schema",
    "estimand",
    "provenance",
)
EXPECTED_DEPTH_AUDIT_SOURCE_HASHES = {
    "audit_decoder_depth_precision.py": EXPECTED_TOOL_HASHES[
        "experiments/transformers/audit_decoder_depth_precision.py"
    ],
    "audit_prompt_numerical_precision.py": EXPECTED_TOOL_HASHES[
        "experiments/transformers/audit_prompt_numerical_precision.py"
    ],
    "finite_basis_pde_icl_campaign.py": EXPECTED_SOURCE_HASHES[
        "finite_basis_pde_icl_campaign.py"
    ],
    "first_principles_decoder_cells.py": EXPECTED_TOOL_HASHES[
        "experiments/transformers/first_principles_decoder_cells.py"
    ],
    "pure_icl_parametric_operator_richardson_attention.py": EXPECTED_SOURCE_HASHES[
        "pure_icl_parametric_operator_richardson_attention.py"
    ],
    "exact_loop_transformer_decoder.py": EXPECTED_SOURCE_HASHES[
        "exact_loop_transformer_decoder.py"
    ],
    "structured_one_head_heavyball.py": EXPECTED_SOURCE_HASHES[
        "structured_one_head_heavyball.py"
    ],
}
EXPECTED_STATIONARY_AUDIT_SOURCE_HASHES = {
    "audit_learned_stationary_decoder.py": EXPECTED_TOOL_HASHES[
        "experiments/transformers/audit_learned_stationary_decoder.py"
    ],
    "audit_decoder_depth_precision.py": EXPECTED_TOOL_HASHES[
        "experiments/transformers/audit_decoder_depth_precision.py"
    ],
    "finite_basis_pde_icl_campaign.py": EXPECTED_SOURCE_HASHES[
        "finite_basis_pde_icl_campaign.py"
    ],
    "pure_icl_parametric_operator_richardson_attention.py": EXPECTED_SOURCE_HASHES[
        "pure_icl_parametric_operator_richardson_attention.py"
    ],
    "exact_loop_transformer_decoder.py": EXPECTED_SOURCE_HASHES[
        "exact_loop_transformer_decoder.py"
    ],
    "structured_one_head_heavyball.py": EXPECTED_SOURCE_HASHES[
        "structured_one_head_heavyball.py"
    ],
}
EXPECTED_FUNCTIONAL_AUDIT_SOURCE_HASHES = {
    "audit_finite_basis_functional_generalization.py": EXPECTED_TOOL_HASHES[
        "experiments/transformers/audit_finite_basis_functional_generalization.py"
    ],
    "audit_prompt_numerical_precision.py": EXPECTED_TOOL_HASHES[
        "experiments/transformers/audit_prompt_numerical_precision.py"
    ],
    "exact_loop_transformer_decoder.py": EXPECTED_SOURCE_HASHES[
        "exact_loop_transformer_decoder.py"
    ],
    "finite_basis_pde_icl_campaign.py": EXPECTED_SOURCE_HASHES[
        "finite_basis_pde_icl_campaign.py"
    ],
    "pure_icl_parametric_operator_richardson_attention.py": EXPECTED_SOURCE_HASHES[
        "pure_icl_parametric_operator_richardson_attention.py"
    ],
    "structured_one_head_heavyball.py": EXPECTED_SOURCE_HASHES[
        "structured_one_head_heavyball.py"
    ],
}
EXPECTED_REPLICA_AUDIT_SOURCE_HASHES = {
    "audit_finite_basis_replica_bounds.py": EXPECTED_TOOL_HASHES[
        "experiments/transformers/audit_finite_basis_replica_bounds.py"
    ],
    "finite_basis_pde_icl_campaign.py": EXPECTED_SOURCE_HASHES[
        "finite_basis_pde_icl_campaign.py"
    ],
    "audit_prompt_numerical_precision.py": EXPECTED_TOOL_HASHES[
        "experiments/transformers/audit_prompt_numerical_precision.py"
    ],
    "pure_icl_parametric_operator_richardson_attention.py": EXPECTED_SOURCE_HASHES[
        "pure_icl_parametric_operator_richardson_attention.py"
    ],
}


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


def verify_toolchain() -> dict[str, str]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PINNED, text=True
    ).strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"pinned worktree moved: {revision}")
    observed = {relative: sha256(PINNED / relative) for relative in EXPECTED_TOOL_HASHES}
    if observed != EXPECTED_TOOL_HASHES:
        raise RuntimeError("pinned manifests or analysis tools changed")
    source_dir = PINNED / "experiments/transformers"
    source_observed = {
        name: sha256(source_dir / name) for name in EXPECTED_SOURCE_HASHES
    }
    if source_observed != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("pinned executable sources changed")
    external_observed = {str(path): sha256(path) for path in EXPECTED_EXTERNAL_HASHES}
    if external_observed != {
        str(path): expected for path, expected in EXPECTED_EXTERNAL_HASHES.items()
    }:
        raise RuntimeError("cohort builder, meta finalizer, or repair manifest changed")
    return {**observed, **source_observed, **external_observed}


def load_campaign_parser():
    path = PINNED / "experiments/transformers/finite_basis_pde_icl_campaign.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("canonical_campaign_225589f", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned campaign parser from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_parser()


def without(mapping: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in excluded}


def finite_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value at {label}")


def finite_tensor_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tensor_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tensor_tree(child, f"{label}[{index}]")
    elif isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite tensor at {label}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite checkpoint scalar at {label}")


def validate_csv(path: Path, *, allow_nonfinite: bool = False) -> int:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row_count = 0
        for row_index, row in enumerate(reader, start=2):
            row_count += 1
            for key, raw in row.items():
                if raw in (None, ""):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not allow_nonfinite and not math.isfinite(value):
                    raise RuntimeError(
                        f"non-finite CSV value at {path}:{row_index}:{key}"
                    )
    if row_count == 0:
        raise RuntimeError(f"empty CSV: {path}")
    return row_count


def load_view(root: Path, phase: str, expected_count: int) -> list[dict[str, Any]]:
    manifest_path = root / "CANONICAL_VIEW_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("runs", [])
    if manifest.get("phase") != phase:
        raise RuntimeError(f"wrong phase in {manifest_path}")
    if manifest.get("expected_runs") != expected_count or len(entries) != expected_count:
        raise RuntimeError(f"wrong count in {manifest_path}")
    if manifest.get("pinned_git_revision") != EXPECTED_REVISION:
        raise RuntimeError(f"wrong revision in {manifest_path}")
    if manifest.get("pinned_source_sha256") != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"wrong source hashes in {manifest_path}")
    names = [str(entry["name"]) for entry in entries]
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate names in {manifest_path}")
    expected_dirs = {(root / name).resolve() for name in names}
    observed_dirs = {marker.parent.resolve() for marker in root.rglob("COMPLETE")}
    if observed_dirs != expected_dirs:
        raise RuntimeError(
            f"unexpected canonical completed directories under {root}: "
            f"missing={sorted(map(str, expected_dirs - observed_dirs))}, "
            f"extra={sorted(map(str, observed_dirs - expected_dirs))}"
        )
    return [dict(entry, phase=phase, canonical_dir=str(root / entry["name"])) for entry in entries]


def check_declared_design(entries: list[dict[str, Any]], parser) -> None:
    primary_cells: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    auxiliary_cells: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for entry in entries:
        expected = vars(
            parser.parse_args(
                ["--outdir", entry["canonical_dir"], *map(str, entry["arguments"])]
            )
        )
        cell = (
            expected["K"],
            expected["active_rank"],
            expected["M0"],
            expected["R"],
            expected["m"],
            expected["probe_layout"],
            expected["loss_profile"],
            expected["init"],
        )
        target = primary_cells if entry["phase"] == "primary" else auxiliary_cells
        target[cell].add(expected["seed"])

    expected_primary = {
        **{
            (4, 4, m0, 8, 128, "leading", "query_only", "true_noisy"): {0, 1, 2}
            for m0 in (2, 4, 8, 16)
        },
        **{
            (8, 8, m0, 8, 128, "leading", "query_only", "true_noisy"): {0, 1, 2}
            for m0 in (4, 8, 16, 32)
        },
        **{
            (8, 8, m0, 16, 64, "leading", "query_only", "true_noisy"): {0, 1}
            for m0 in (4, 8, 16, 32)
        },
        **{
            (8, 8, m0, 8, 128, "evenly_spaced", "query_only", "true_noisy"): {0, 1, 2}
            for m0 in (4, 8, 16, 32)
        },
    }
    expected_auxiliary = {
        (4, 4, 2, 8, 128, "leading", "query_only", "true_A0_random_basis"): {0, 1, 2},
        (4, 4, 8, 8, 128, "leading", "query_only", "true_A0_random_basis"): {0, 1, 2},
        **{
            (4, 4, 8, 8, 128, "leading", objective, "true_noisy"): {0, 1, 2}
            for objective in (
                "query_prompt",
                "query_operator",
                "query_coefficient",
                "query_latent",
            )
        },
    }
    if dict(primary_cells) != expected_primary:
        raise RuntimeError("primary scientific cells or seed coverage differ from design")
    if dict(auxiliary_cells) != expected_auxiliary:
        raise RuntimeError("auxiliary scientific cells or seed coverage differ from design")


def validate_provenance(provenance: dict[str, Any], run_dir: Path) -> None:
    if provenance.get("schema_version") != 1:
        raise RuntimeError(f"wrong checkpoint provenance schema in {run_dir}")
    for key in ("git_revision", "run_origin_git_revision"):
        if provenance.get(key) != EXPECTED_REVISION:
            raise RuntimeError(f"wrong {key} in {run_dir}")
    for key in ("campaign_source_sha256", "run_origin_campaign_source_sha256"):
        if provenance.get(key) != EXPECTED_SOURCE_HASHES[
            "finite_basis_pde_icl_campaign.py"
        ]:
            raise RuntimeError(f"wrong {key} in {run_dir}")
    for key in ("source_sha256", "run_origin_source_sha256"):
        if provenance.get(key) != EXPECTED_SOURCE_HASHES:
            raise RuntimeError(f"wrong {key} in {run_dir}")


def validate_run(
    entry: dict[str, Any], parser
) -> tuple[dict[str, Any], dict[str, str]]:
    run_dir = Path(entry["canonical_dir"]).resolve()
    for name in (*RAW_ARTIFACTS, *PLOT_ARTIFACTS, "reconstruction_examples.pt"):
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing canonical artifact: {path}")
    run = json.loads((run_dir / "run.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    finite_tree(run, str(run_dir / "run.json"))
    finite_tree(summary, str(run_dir / "summary.json"))
    if run.get("status") != "complete":
        raise RuntimeError(f"run is not complete: {run_dir}")
    expected = vars(
        parser.parse_args(
            ["--outdir", str(run_dir), *map(str, entry["arguments"])]
        )
    )
    saved_args = dict(run.get("args", {}))
    if without(saved_args, ARGUMENT_EXCLUSIONS) != without(
        expected, ARGUMENT_EXCLUSIONS
    ):
        raise RuntimeError(f"full resolved argument mismatch: {run_dir}")
    if run.get("git_revision_at_start") != EXPECTED_REVISION:
        raise RuntimeError(f"wrong initial revision: {run_dir}")
    if run.get("source_sha256_at_initial_start") != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"wrong initial source hashes: {run_dir}")
    if run.get("source_sha256") != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"wrong final source hashes: {run_dir}")

    checkpoint_path = run_dir / "model_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    finite_tensor_tree(checkpoint, str(checkpoint_path))
    checkpoint_args = dict(checkpoint.get("args", {}))
    if checkpoint_args != saved_args:
        raise RuntimeError(f"checkpoint/run argument mismatch: {run_dir}")
    if without(checkpoint_args, ARGUMENT_EXCLUSIONS) != without(
        expected, ARGUMENT_EXCLUSIONS
    ):
        raise RuntimeError(f"checkpoint/manifest argument mismatch: {run_dir}")
    if int(checkpoint.get("step", -1)) != int(expected["steps"]):
        raise RuntimeError(f"checkpoint is not at the configured final step: {run_dir}")
    provenance = checkpoint.get("campaign_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"checkpoint lacks strict provenance: {run_dir}")
    validate_provenance(provenance, run_dir)
    if run.get("checkpoint_provenance") != provenance:
        raise RuntimeError(f"run/checkpoint provenance mismatch: {run_dir}")

    bank = torch.load(run_dir / "task_bank.pt", map_location="cpu", weights_only=True)
    finite_tensor_tree(bank, str(run_dir / "task_bank.pt"))
    reconstruction = torch.load(
        run_dir / "reconstruction_examples.pt", map_location="cpu", weights_only=True
    )
    finite_tensor_tree(
        reconstruction, str(run_dir / "reconstruction_examples.pt")
    )
    for key in ("train_z", "heldout_z"):
        if key not in checkpoint or key not in bank:
            raise RuntimeError(f"missing embedded {key}: {run_dir}")
        if not torch.equal(checkpoint[key].cpu(), bank[key].cpu()):
            raise RuntimeError(f"checkpoint/task-bank {key} mismatch: {run_dir}")
        if not torch.isfinite(bank[key]).all():
            raise RuntimeError(f"non-finite task-bank {key}: {run_dir}")

    csv_rows = {
        name: validate_csv(run_dir / name)
        for name in (
            "train.csv",
            "evaluation.csv",
            "decoder_depth.csv",
        )
    }
    if csv_rows["decoder_depth.csv"] != 40:
        raise RuntimeError(f"decoder depth grid is incomplete: {run_dir}")

    artifact_hashes = {
        name: sha256(run_dir / name)
        for name in (*RAW_ARTIFACTS, *PLOT_ARTIFACTS, "reconstruction_examples.pt")
    }
    checkpoint_hash = sha256(checkpoint_path)
    return expected, {"model_final.pt": checkpoint_hash, **artifact_hashes}


def expected_paired_run_metrics(
    entries: list[dict[str, Any]], resolved: dict[str, dict[str, Any]]
) -> set[tuple[str, str, str]]:
    by_design: dict[tuple[Any, ...], dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in entries:
        run_dir = str(Path(entry["canonical_dir"]).resolve())
        args = resolved[run_dir]
        if args["K"] != 4 or args["M0"] not in (2, 8):
            continue
        design = (
            args["K"],
            args["active_rank"],
            args["M0"],
            args["R"],
            args["m"],
            args["depth"],
            args["d"],
            args["steps"],
        )
        by_design[design][args["seed"]].append(run_dir)
    expected: set[tuple[str, str, str]] = set()
    for seed_groups in by_design.values():
        for runs in seed_groups.values():
            for left_index, run_a in enumerate(sorted(runs)):
                for run_b in sorted(runs)[left_index + 1 :]:
                    for metric in METRIC_FIELDS:
                        expected.add((run_a, run_b, metric))
    return expected


def validate_meta(
    entries: list[dict[str, Any]],
    checkpoint_hashes: dict[str, str],
    resolved: dict[str, dict[str, Any]],
) -> dict[str, int]:
    payload = json.loads((META / "meta_analysis.json").read_text())
    finite_tree(payload, str(META / "meta_analysis.json"))
    checkpoints = payload.get("checkpoints", [])
    differences = payload.get("matched_differences", [])
    summaries = payload.get("matched_summary", [])
    if payload.get("issues"):
        raise RuntimeError("canonical meta-analysis reports issues")
    expected_dirs = {str(Path(entry["canonical_dir"]).resolve()) for entry in entries}
    observed_dirs = {str(Path(row["run_dir"]).resolve()) for row in checkpoints}
    if len(checkpoints) != 62 or observed_dirs != expected_dirs:
        raise RuntimeError("meta checkpoint set is not the exact manifest union")
    if len(observed_dirs) != len(checkpoints):
        raise RuntimeError("meta checkpoint rows contain duplicates")
    names = [str(row.get("run")) for row in checkpoints]
    hashes = [str(row.get("checkpoint_sha256")) for row in checkpoints]
    if len(names) != len(set(names)) or len(hashes) != len(set(hashes)):
        raise RuntimeError("meta run names or checkpoint hashes are not unique")
    for row in checkpoints:
        run_dir = str(Path(row["run_dir"]).resolve())
        if str(row.get("checkpoint_kind")) != "final":
            raise RuntimeError(f"meta did not select final checkpoint: {run_dir}")
        if str(row.get("checkpoint_sha256")) != checkpoint_hashes[run_dir]:
            raise RuntimeError(f"meta checkpoint hash mismatch: {run_dir}")
        if int(row.get("checkpoint_step", -1)) != int(resolved[run_dir]["steps"]):
            raise RuntimeError(f"meta checkpoint step mismatch: {run_dir}")
        for metric, expected_provenance in EXPECTED_METRIC_PROVENANCE.items():
            observed_provenance = tuple(
                str(row.get(f"{metric}_{field}"))
                for field in ("source", "schema", "provenance")
            )
            if observed_provenance != expected_provenance:
                raise RuntimeError(
                    f"meta {metric} provenance changed in {run_dir}: "
                    f"{observed_provenance} != {expected_provenance}"
                )
            if str(row.get(f"{metric}_checkpoint_sha256")) != checkpoint_hashes[
                run_dir
            ]:
                raise RuntimeError(f"meta {metric} checkpoint binding failed: {run_dir}")
            artifact = Path(str(row.get(f"{metric}_artifact"))).resolve()
            if not artifact.is_file():
                raise RuntimeError(f"meta {metric} artifact is missing: {artifact}")

    observed_pairs = {
        (
            *sorted(
                (
                    str(Path(row["run_a"]).resolve()),
                    str(Path(row["run_b"]).resolve()),
                )
            ),
            str(row["metric"]),
        )
        for row in differences
    }
    expected_pairs = expected_paired_run_metrics(entries, resolved)
    if observed_pairs != expected_pairs or len(differences) != len(expected_pairs):
        raise RuntimeError(
            f"matched comparison set differs from design: "
            f"expected={len(expected_pairs)} observed={len(differences)}"
        )
    if len(expected_pairs) != 288:
        raise RuntimeError(f"internal expected matched metric count is {len(expected_pairs)}")

    grouped_differences: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in differences:
        grouped_differences[
            tuple(row.get(field) for field in MATCHED_SUMMARY_FIELDS)
        ].append(row)
    summary_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in summaries:
        key = tuple(row.get(field) for field in MATCHED_SUMMARY_FIELDS)
        if key in summary_by_key:
            raise RuntimeError("duplicate matched-summary grouping key")
        summary_by_key[key] = row
    if set(summary_by_key) != set(grouped_differences):
        raise RuntimeError("matched-summary grouping set differs from exact differences")
    if len(summaries) != 96:
        raise RuntimeError(f"expected 96 seed-aggregated comparisons, got {len(summaries)}")

    def mean_and_se(values: list[float]) -> tuple[float, float]:
        mean = sum(values) / len(values)
        if len(values) == 1:
            return mean, 0.0
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return mean, math.sqrt(variance) / math.sqrt(len(values))

    for key, group in grouped_differences.items():
        summary = summary_by_key[key]
        if int(summary.get("paired_checkpoints", -1)) != len(group) or len(group) != 3:
            raise RuntimeError("a matched summary lacks all three paired checkpoints")
        differences_values = [float(row["difference_b_minus_a"]) for row in group]
        expected_mean, expected_se = mean_and_se(differences_values)
        for field, expected_value in (
            ("mean_difference_b_minus_a", expected_mean),
            ("se_difference_b_minus_a", expected_se),
        ):
            if not math.isclose(
                float(summary[field]), expected_value, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise RuntimeError(f"matched-summary aggregate mismatch: {field}")
        ratios = [
            float(row["log10_ratio_b_over_a"])
            for row in group
            if row.get("log10_ratio_b_over_a") is not None
        ]
        if ratios:
            expected_mean, expected_se = mean_and_se(ratios)
            for field, expected_value in (
                ("mean_log10_ratio_b_over_a", expected_mean),
                ("se_log10_ratio_b_over_a", expected_se),
            ):
                if not math.isclose(
                    float(summary[field]), expected_value, rel_tol=1e-12, abs_tol=1e-15
                ):
                    raise RuntimeError(f"matched-summary aggregate mismatch: {field}")
        elif summary.get("mean_log10_ratio_b_over_a") is not None:
            raise RuntimeError("matched-summary unexpectedly reports a log ratio")
    return {
        "checkpoint_rows": len(checkpoints),
        "checkpoint_names": len(set(names)),
        "checkpoint_hashes": len(set(hashes)),
        "paired_metric_rows": len(differences),
        "seed_aggregated_comparisons": len(summaries),
    }


def validate_root_audits(
    root: Path,
    entries: list[dict[str, Any]],
    checkpoint_hashes: dict[str, str],
) -> dict[str, str]:
    expected_dirs = {
        str(Path(entry["canonical_dir"]).resolve()) for entry in entries
    }
    required_root_artifacts = (
        "functional_generalization_audit.json",
        "functional_generalization_samples.csv",
        "functional_generalization.png",
        "functional_generalization.pdf",
        "bound_and_observability_audit.json",
        "bound_and_observability_audit.csv",
    )
    for name in required_root_artifacts:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing root audit artifact: {path}")
    # Raw solve samples intentionally retain NaN on a failed numerical solve;
    # the schema-2 JSON below must account for them via finite counts and
    # failure rates. Aggregates and every other canonical CSV remain finite.
    validate_csv(
        root / "functional_generalization_samples.csv", allow_nonfinite=True
    )
    validate_csv(root / "bound_and_observability_audit.csv")
    functional = json.loads(
        (root / "functional_generalization_audit.json").read_text()
    )
    finite_tree(functional, str(root / "functional_generalization_audit.json"))
    if functional.get("schema_version") != 2 or functional.get("skipped_runs"):
        raise RuntimeError(f"root functional audit incomplete: {root}")
    if functional.get("provenance", {}).get(
        "source_sha256"
    ) != EXPECTED_FUNCTIONAL_AUDIT_SOURCE_HASHES:
        raise RuntimeError(f"root functional audit source mismatch: {root}")
    functional_artifacts = functional.get("provenance", {}).get("run_artifacts", [])
    observed = {str(Path(row["run"]).resolve()) for row in functional_artifacts}
    if observed != expected_dirs:
        raise RuntimeError(f"root functional audit run set mismatch: {root}")
    if len(functional_artifacts) != len(entries):
        raise RuntimeError(f"root functional audit has duplicate runs: {root}")
    for row in functional_artifacts:
        run_dir = str(Path(row["run"]).resolve())
        if str(row.get("checkpoint_sha256")) != checkpoint_hashes[run_dir]:
            raise RuntimeError(f"root functional checkpoint hash mismatch: {run_dir}")
        if str(row.get("task_bank_sha256")) != sha256(
            Path(run_dir) / "task_bank.pt"
        ):
            raise RuntimeError(f"root functional task-bank hash mismatch: {run_dir}")
        if not row.get("checkpoint_embeds_heldout_z") or not row.get(
            "embedded_heldout_z_matches_task_bank"
        ):
            raise RuntimeError(f"root functional heldout pairing failed: {run_dir}")

    replica = json.loads(
        (root / "bound_and_observability_audit.json").read_text()
    )
    finite_tree(replica, str(root / "bound_and_observability_audit.json"))
    replica_dirs = {str(Path(row["run"]).resolve()) for row in replica.get("records", [])}
    if replica.get("schema_version") != 2 or replica.get("run_count") != len(entries):
        raise RuntimeError(f"root replica audit count mismatch: {root}")
    if replica.get("audit_config", {}).get(
        "source_sha256"
    ) != EXPECTED_REPLICA_AUDIT_SOURCE_HASHES:
        raise RuntimeError(f"root replica audit source mismatch: {root}")
    if replica_dirs != expected_dirs:
        raise RuntimeError(f"root replica audit run set mismatch: {root}")
    for row in replica.get("records", []):
        run_dir = str(Path(row["run"]).resolve())
        if row.get("audit_config", {}).get(
            "source_sha256"
        ) != EXPECTED_REPLICA_AUDIT_SOURCE_HASHES:
            raise RuntimeError(f"replica per-run source mismatch: {run_dir}")
        artifacts = row.get("audit_config", {}).get("artifact_sha256", {})
        if str(artifacts.get("model_final.pt")) != checkpoint_hashes[run_dir]:
            raise RuntimeError(f"replica checkpoint hash mismatch: {run_dir}")
        if str(artifacts.get("task_bank.pt")) != sha256(
            Path(run_dir) / "task_bank.pt"
        ):
            raise RuntimeError(f"replica task-bank hash mismatch: {run_dir}")
        if str(artifacts.get("run.json")) != sha256(Path(run_dir) / "run.json"):
            raise RuntimeError(f"replica run-metadata hash mismatch: {run_dir}")
    depth_dir = root / "decoder_depth_precision_audit"
    depth_summary_path = depth_dir / "summary.json"
    depth_csv_path = depth_dir / "decoder_depth_precision.csv"
    if not depth_summary_path.is_file() or not depth_csv_path.is_file():
        raise RuntimeError(f"missing decoder-depth audit outputs: {depth_dir}")
    validate_csv(depth_csv_path)
    depth = json.loads(depth_summary_path.read_text())
    finite_tree(depth, str(depth_summary_path))
    if depth.get("status") != "complete" or depth.get("run_count") != len(entries):
        raise RuntimeError(f"decoder-depth audit count/status mismatch: {root}")
    if depth.get("source_sha256") != EXPECTED_DEPTH_AUDIT_SOURCE_HASHES:
        raise RuntimeError(f"decoder-depth audit source mismatch: {root}")
    depth_runs = depth.get("runs", [])
    if len(depth_runs) != len(entries):
        raise RuntimeError(f"decoder-depth audit duplicate/missing runs: {root}")
    if {str(Path(row["run_dir"]).resolve()) for row in depth_runs} != expected_dirs:
        raise RuntimeError(f"decoder-depth audit run set mismatch: {root}")
    for row in depth_runs:
        run_dir = str(Path(row["run_dir"]).resolve())
        if str(row.get("checkpoint_sha256")) != checkpoint_hashes[run_dir]:
            raise RuntimeError(f"decoder-depth checkpoint hash mismatch: {run_dir}")
        if row.get("source_sha256") != EXPECTED_DEPTH_AUDIT_SOURCE_HASHES:
            raise RuntimeError(f"decoder-depth per-run source mismatch: {run_dir}")
        if str(row.get("task_bank_sha256")) != sha256(
            Path(run_dir) / "task_bank.pt"
        ):
            raise RuntimeError(f"decoder-depth task-bank hash mismatch: {run_dir}")

    stationary_dir = root / "learned_stationary_decoder_audit"
    stationary_json_path = stationary_dir / "learned_stationary_decoder_audit.json"
    stationary_csv_path = stationary_dir / "learned_stationary_decoder_audit.csv"
    if not stationary_json_path.is_file() or not stationary_csv_path.is_file():
        raise RuntimeError(f"missing stationary audit outputs: {stationary_dir}")
    validate_csv(stationary_csv_path)
    stationary = json.loads(stationary_json_path.read_text())
    finite_tree(stationary, str(stationary_json_path))
    if (
        stationary.get("status") != "complete"
        or stationary.get("completed_checkpoint_count") != len(entries)
        or stationary.get("compatible_checkpoint_count") != len(entries)
    ):
        raise RuntimeError(f"stationary audit count/status mismatch: {root}")
    if stationary.get("source_sha256") != EXPECTED_STATIONARY_AUDIT_SOURCE_HASHES:
        raise RuntimeError(f"stationary audit source mismatch: {root}")
    stationary_results = stationary.get("results", [])
    if len(stationary_results) != len(entries):
        raise RuntimeError(f"stationary audit duplicate/missing runs: {root}")
    if {
        str(Path(row["run_dir"]).resolve()) for row in stationary_results
    } != expected_dirs:
        raise RuntimeError(f"stationary audit run set mismatch: {root}")
    for row in stationary_results:
        run_dir = str(Path(row["run_dir"]).resolve())
        if row.get("status") != "audited":
            raise RuntimeError(f"stationary audit did not audit checkpoint: {run_dir}")
        if str(row.get("checkpoint_sha256")) != checkpoint_hashes[run_dir]:
            raise RuntimeError(f"stationary checkpoint hash mismatch: {run_dir}")
        if row.get("source_sha256") != EXPECTED_STATIONARY_AUDIT_SOURCE_HASHES:
            raise RuntimeError(f"stationary per-run source mismatch: {run_dir}")
        if str(row.get("task_bank_sha256")) != sha256(
            Path(run_dir) / "task_bank.pt"
        ):
            raise RuntimeError(f"stationary task-bank hash mismatch: {run_dir}")

    return {
        **{name: sha256(root / name) for name in required_root_artifacts},
        "decoder_depth_precision_audit/summary.json": sha256(depth_summary_path),
        "decoder_depth_precision_audit/decoder_depth_precision.csv": sha256(
            depth_csv_path
        ),
        "learned_stationary_decoder_audit/learned_stationary_decoder_audit.json": sha256(
            stationary_json_path
        ),
        "learned_stationary_decoder_audit/learned_stationary_decoder_audit.csv": sha256(
            stationary_csv_path
        ),
    }


def main() -> None:
    if CERTIFIED.exists():
        raise RuntimeError(f"one-shot publication certificate already exists: {CERTIFIED}")
    wait_for_all((META_READY, PLOTS_READY))
    toolchain_hashes = verify_toolchain()
    parser = load_campaign_parser()
    primary = load_view(PRIMARY, "primary", 44)
    auxiliary = load_view(AUXILIARY, "auxiliary", 18)
    entries = [*primary, *auxiliary]
    if len(entries) != 62 or len({entry["name"] for entry in entries}) != 62:
        raise RuntimeError("canonical views do not contain 62 globally unique names")
    check_declared_design(entries, parser)

    resolved: dict[str, dict[str, Any]] = {}
    checkpoint_hashes: dict[str, str] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    for index, entry in enumerate(entries, start=1):
        print(f"CERTIFYING {index}/62 {entry['name']}", flush=True)
        expected, artifacts = validate_run(entry, parser)
        run_dir = str(Path(entry["canonical_dir"]).resolve())
        resolved[run_dir] = expected
        checkpoint_hashes[run_dir] = artifacts["model_final.pt"]
        artifact_hashes[run_dir] = artifacts
    if len(set(checkpoint_hashes.values())) != 62:
        raise RuntimeError("canonical model_final checkpoints are not unique")

    root_audit_hashes = {
        "primary": validate_root_audits(PRIMARY, primary, checkpoint_hashes),
        "auxiliary": validate_root_audits(AUXILIARY, auxiliary, checkpoint_hashes),
    }
    meta_validation = validate_meta(entries, checkpoint_hashes, resolved)
    plots_certificate = json.loads(PLOTS_READY.read_text())
    if plots_certificate.get("status") != "complete":
        raise RuntimeError("plot refresh certificate is not complete")
    if plots_certificate.get("run_count") != 62:
        raise RuntimeError("plot refresh certificate does not cover 62 runs")
    if plots_certificate.get("pinned_revision") != EXPECTED_REVISION:
        raise RuntimeError("plot refresh certificate has the wrong pinned revision")
    if plots_certificate.get("plotter_sha256") != EXPECTED_TOOL_HASHES[
        "experiments/transformers/plot_finite_basis_pde_campaign.py"
    ]:
        raise RuntimeError("plot refresh certificate has the wrong plotter hash")
    if plots_certificate.get("executable_source_sha256") != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("plot refresh certificate has the wrong source hashes")
    expected_plot_dirs = set(checkpoint_hashes)
    certified_plots = plots_certificate.get("per_run_artifact_sha256", {})
    if set(certified_plots) != expected_plot_dirs:
        raise RuntimeError("plot refresh certificate run set mismatch")
    for run_dir, certified_hashes in certified_plots.items():
        if set(certified_hashes) != FRESH_PLOT_ARTIFACTS:
            raise RuntimeError(f"plot refresh artifact set mismatch: {run_dir}")
        for name, expected_hash in certified_hashes.items():
            if artifact_hashes[run_dir].get(name) != expected_hash:
                raise RuntimeError(f"fresh plot hash changed: {run_dir}/{name}")
    certified_aggregates = plots_certificate.get("aggregate_artifact_sha256", {})
    if set(certified_aggregates) != {str(PRIMARY.resolve()), str(AUXILIARY.resolve())}:
        raise RuntimeError("plot refresh aggregate roots mismatch")
    for root, hashes in certified_aggregates.items():
        if set(hashes) != AGGREGATE_ARTIFACTS:
            raise RuntimeError(f"aggregate artifact set mismatch: {root}")
        for name, expected_hash in hashes.items():
            if sha256(Path(root) / name) != expected_hash:
                raise RuntimeError(f"fresh aggregate hash changed: {root}/{name}")

    meta_certificate = json.loads(META_READY.read_text())
    expected_meta_artifacts = {
        "meta_analysis.json",
        "REPORT.md",
        "checkpoint_metrics.csv",
        "matched_differences.csv",
        "matched_summary.csv",
        "checkpoint_overview.png",
        "checkpoint_overview.pdf",
        "matched_comparisons.png",
        "matched_comparisons.pdf",
    }
    certified_meta_artifacts = meta_certificate.get("artifacts", {})
    if set(certified_meta_artifacts) != expected_meta_artifacts:
        raise RuntimeError("canonical meta certificate artifact set mismatch")
    for name, expected_hash in certified_meta_artifacts.items():
        if sha256(META / name) != expected_hash:
            raise RuntimeError(f"canonical meta artifact hash changed: {name}")
    verify_toolchain()
    payload = {
        "status": "certified",
        "created_unix": time.time(),
        "statement": (
            "Exactly 62 manifest-declared, revision-225589f checkpoints passed full "
            "argument, checkpoint/task-bank, source, audit, comparison, and pinned-plot checks."
        ),
        "pinned_revision": EXPECTED_REVISION,
        "primary_runs": len(primary),
        "auxiliary_runs": len(auxiliary),
        "total_runs": len(entries),
        "source_sha256": EXPECTED_SOURCE_HASHES,
        "toolchain_sha256": toolchain_hashes,
        "meta_validation": meta_validation,
        "plot_certificate_sha256": sha256(PLOTS_READY),
        "canonical_meta_certificate_sha256": sha256(META_READY),
        "canonical_view_manifest_sha256": {
            "primary": sha256(PRIMARY / "CANONICAL_VIEW_MANIFEST.json"),
            "auxiliary": sha256(AUXILIARY / "CANONICAL_VIEW_MANIFEST.json"),
        },
        "root_audit_artifact_sha256": root_audit_hashes,
        "per_run_artifact_sha256": artifact_hashes,
    }
    atomic_json(CERTIFIED, payload)
    print(f"PUBLICATION COHORT CERTIFIED {CERTIFIED}", flush=True)


if __name__ == "__main__":
    main()

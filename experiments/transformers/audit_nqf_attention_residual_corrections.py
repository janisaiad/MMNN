"""Audit NQF orders for residual-correction attention blocks.

This experiment separates two parameterizations that have different local
normal forms:

* full multi-head attention, with Q/K/V/readout all initialized at scale eps;
* query-key-only attention, with a fixed nonzero value/readout vector.

It measures output Taylor orders and loss-gradient orders near the origin.
The experiment is deliberately local: it diagnoses which blocks receive a
residual-correction learning signal at initialization.  It does not identify
the later PCG Krylov polynomial and does not test a full PDE solve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .low_rank_subspace_preconditioner import qk_only_routing_nqf
except ImportError:
    from low_rank_subspace_preconditioner import qk_only_routing_nqf

Tensor = torch.Tensor


@dataclass(frozen=True)
class AuditConfig:
    seed: int = 260813335
    batch_size: int = 256
    input_dimension: int = 7
    sequence_length: int = 11
    heads: int = 3
    key_dimension: int = 5
    value_dimension: int = 6
    slots: int = 4
    epsilon_min_power: float = -4.0
    epsilon_max_power: float = -0.7
    epsilon_count: int = 12


def _unit_frobenius(shape: Tuple[int, ...], generator: torch.Generator) -> Tensor:
    value = torch.randn(*shape, generator=generator, dtype=torch.float64)
    return value / torch.linalg.vector_norm(value)


def _full_mha(
    context: Tensor,
    query: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    readout: Tensor,
    uniform: bool = False,
) -> Tensor:
    projected_query = torch.einsum("hkd,bd->bhk", q_weight, query)
    projected_keys = torch.einsum("hkd,bdn->bhkn", k_weight, context)
    logits = torch.einsum(
        "bhk,bhkn->bhn",
        projected_query,
        projected_keys,
    ) / math.sqrt(q_weight.shape[1])
    if uniform:
        attention = torch.full_like(logits, 1.0 / context.shape[-1])
    else:
        attention = torch.softmax(logits, dim=-1)
    values = torch.einsum("hvd,bdn->bhvn", v_weight, context)
    pooled = torch.einsum("bhn,bhvn->bhv", attention, values)
    return torch.einsum("hv,bhv->b", readout, pooled)


def _qk_only_attention(
    context: Tensor,
    query: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    fixed_readout: Tensor,
    uniform: bool = False,
) -> Tensor:
    projected_query = torch.einsum("hkd,bd->bhk", q_weight, query)
    projected_keys = torch.einsum("hkd,bdn->bhkn", k_weight, context)
    logits = torch.einsum(
        "bhk,bhkn->bhn",
        projected_query,
        projected_keys,
    ) / math.sqrt(q_weight.shape[1])
    if uniform:
        attention = torch.full_like(logits, 1.0 / context.shape[-1])
    else:
        attention = torch.softmax(logits, dim=-1)
    fixed_values = torch.einsum("hd,bdn->bhn", fixed_readout, context)
    return torch.einsum("bhn,bhn->b", fixed_values, attention)


def _gradient_norm(gradients: Iterable[Tensor]) -> float:
    squared = sum(float(gradient.square().sum()) for gradient in gradients)
    return math.sqrt(squared)


def _rms(value: Tensor) -> float:
    return float(value.square().mean().sqrt())


def _fit_log_slope(rows: list[Dict[str, float]], key: str) -> float:
    epsilon = np.asarray([row["epsilon"] for row in rows])
    values = np.asarray([row[key] for row in rows])
    # Subtractions used to isolate a Taylor remainder eventually reach the
    # float64 noise floor.  Fit only the resolved part of each curve.
    resolved_floor = float(values.max()) * 1e-8
    valid = np.isfinite(values) & (values > max(1e-28, resolved_floor))
    return float(np.polyfit(np.log(epsilon[valid]), np.log(values[valid]), 1)[0])


def run_audit(config: AuditConfig) -> Tuple[list[Dict[str, float]], Dict[str, object]]:
    generator = torch.Generator().manual_seed(config.seed)
    batch = config.batch_size
    dimension = config.input_dimension
    context = torch.randn(
        batch,
        dimension,
        config.sequence_length,
        generator=generator,
        dtype=torch.float64,
    )
    query = torch.randn(batch, dimension, generator=generator, dtype=torch.float64)
    target = torch.randn(batch, generator=generator, dtype=torch.float64)
    target = target / target.square().mean().sqrt()

    q_base = _unit_frobenius((config.heads, config.key_dimension, dimension), generator)
    k_base = _unit_frobenius((config.heads, config.key_dimension, dimension), generator)
    v_base = _unit_frobenius(
        (config.heads, config.value_dimension, dimension), generator
    )
    readout_base = _unit_frobenius((config.heads, config.value_dimension), generator)
    fixed_readout = _unit_frobenius((config.heads, dimension), generator)

    normalized_rows = context.transpose(-1, -2)
    normalized_rows = normalized_rows / normalized_rows.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    slot_query_base = _unit_frobenius((config.slots, config.key_dimension), generator)
    slot_key_base = _unit_frobenius((config.key_dimension, dimension), generator)

    rows: list[Dict[str, float]] = []
    epsilons = np.logspace(
        config.epsilon_min_power,
        config.epsilon_max_power,
        config.epsilon_count,
    )
    for epsilon_value in epsilons:
        epsilon = float(epsilon_value)
        q = (epsilon * q_base).clone().requires_grad_(True)
        k = (epsilon * k_base).clone().requires_grad_(True)
        v = (epsilon * v_base).clone().requires_grad_(True)
        readout = (epsilon * readout_base).clone().requires_grad_(True)
        full_output = _full_mha(context, query, q, k, v, readout)
        full_uniform = _full_mha(context, query, q, k, v, readout, uniform=True)
        full_loss = 0.5 * (full_output - target).square().mean()
        full_gradients = torch.autograd.grad(full_loss, (q, k, v, readout))

        q_only = (epsilon * q_base).clone().requires_grad_(True)
        k_only = (epsilon * k_base).clone().requires_grad_(True)
        qk_output = _qk_only_attention(
            context,
            query,
            q_only,
            k_only,
            fixed_readout,
        )
        qk_uniform = _qk_only_attention(
            context,
            query,
            q_only,
            k_only,
            fixed_readout,
            uniform=True,
        )
        qk_loss = 0.5 * (qk_output - target).square().mean()
        qk_gradients = torch.autograd.grad(qk_loss, (q_only, k_only))

        slot_queries = epsilon * slot_query_base
        slot_key = epsilon * slot_key_base
        slot_keys = torch.einsum("hd,bmd->bmh", slot_key, normalized_rows)
        slot_logits = torch.einsum("sh,bmh->bsm", slot_queries, slot_keys) / math.sqrt(
            config.key_dimension
        )
        slot_attention = torch.softmax(slot_logits, dim=-1)
        routed_rows = torch.einsum("bsm,bmd->bsd", slot_attention, normalized_rows)
        routed_nqf = qk_only_routing_nqf(
            normalized_rows,
            slot_key,
            slot_queries,
        )
        uniform_rows = normalized_rows.mean(dim=1, keepdim=True)

        rows.append(
            {
                "epsilon": epsilon,
                "full_total_output_rms": _rms(full_output),
                "full_routing_output_rms": _rms(full_output - full_uniform),
                "full_qk_loss_gradient_norm": _gradient_norm(full_gradients[:2]),
                "full_vo_loss_gradient_norm": _gradient_norm(full_gradients[2:]),
                "qk_only_routing_output_rms": _rms(qk_output - qk_uniform),
                "qk_only_loss_gradient_norm": _gradient_norm(qk_gradients),
                "decoder_pre_qr_routing_rms": _rms(routed_rows - uniform_rows),
                "decoder_pre_qr_nqf_remainder_rms": _rms(routed_rows - routed_nqf),
            }
        )

    expectations = {
        "full_total_output_rms": 2.0,
        "full_routing_output_rms": 4.0,
        "full_qk_loss_gradient_norm": 3.0,
        "full_vo_loss_gradient_norm": 1.0,
        "qk_only_routing_output_rms": 2.0,
        "qk_only_loss_gradient_norm": 1.0,
        "decoder_pre_qr_routing_rms": 2.0,
        "decoder_pre_qr_nqf_remainder_rms": 4.0,
    }
    slopes = {key: _fit_log_slope(rows, key) for key in expectations}
    tolerance = 0.2
    checks = {
        key: abs(slopes[key] - expected) <= tolerance
        for key, expected in expectations.items()
    }
    smallest = rows[0]
    summary: Dict[str, object] = {
        "config": asdict(config),
        "expected_slopes": expectations,
        "measured_slopes": slopes,
        "slope_tolerance": tolerance,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "smallest_epsilon_full_qk_to_vo_gradient_ratio": (
            smallest["full_qk_loss_gradient_norm"]
            / smallest["full_vo_loss_gradient_norm"]
        ),
        "smallest_epsilon_qk_only_to_full_qk_gradient_ratio": (
            smallest["qk_only_loss_gradient_norm"]
            / smallest["full_qk_loss_gradient_norm"]
        ),
        "interpretation": {
            "full_mha": (
                "V/readout receive the quadratic-order residual signal; "
                "Q/K routing is delayed to quartic output order when all "
                "four blocks are small."
            ),
            "query_key_only": (
                "With fixed nonzero values/readout, Q/K routing is itself "
                "quadratic and receives an order-epsilon loss gradient."
            ),
            "decoder_head": (
                "The fixed-value pre-QR routing in OneHeadObservableSubspace "
                "has the query-key-only NQF. QR/Cholesky/PCG are outside this "
                "local smooth normal form and remain exact primitives."
            ),
        },
    }
    return rows, summary


def write_outputs(
    rows: list[Dict[str, float]],
    summary: Dict[str, object],
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_directory / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    epsilon = np.asarray([row["epsilon"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for key, label in [
        ("full_total_output_rms", "full MHA: total"),
        ("full_routing_output_rms", "full MHA: routed part"),
        ("qk_only_routing_output_rms", "QK-only: routed part"),
        ("decoder_pre_qr_nqf_remainder_rms", "decoder pre-QR: NQF remainder"),
    ]:
        axes[0].loglog(epsilon, [row[key] for row in rows], "o-", label=label)
    axes[0].set_xlabel("initialization scale $\\epsilon$")
    axes[0].set_ylabel("RMS magnitude")
    axes[0].set_title("Output Taylor orders")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    for key, label in [
        ("full_vo_loss_gradient_norm", "full MHA: V/O gradient"),
        ("full_qk_loss_gradient_norm", "full MHA: Q/K gradient"),
        ("qk_only_loss_gradient_norm", "QK-only: Q/K gradient"),
    ]:
        axes[1].loglog(epsilon, [row[key] for row in rows], "o-", label=label)
    axes[1].set_xlabel("initialization scale $\\epsilon$")
    axes[1].set_ylabel("loss-gradient norm")
    axes[1].set_title("Residual-correction trainability")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_directory / "nqf_attention_orders.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/transformers/nqf_attention_residual_audit"),
    )
    args = parser.parse_args()
    rows, summary = run_audit(AuditConfig())
    write_outputs(rows, summary, args.outdir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_checks_pass"]:
        raise SystemExit("one or more NQF order checks failed")


if __name__ == "__main__":
    main()

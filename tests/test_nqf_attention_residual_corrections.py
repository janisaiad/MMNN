import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PRECONDITIONER = _load(
    "low_rank_subspace_preconditioner",
    "experiments/transformers/low_rank_subspace_preconditioner.py",
)
AUDIT = _load(
    "audit_nqf_attention_residual_corrections_test",
    "experiments/transformers/audit_nqf_attention_residual_corrections.py",
)


def test_query_key_routing_normal_form_has_quartic_remainder() -> None:
    torch.manual_seed(13335)
    rows = torch.randn(8, 9, 6, dtype=torch.float64)
    rows = rows / rows.norm(dim=-1, keepdim=True)
    key = torch.randn(5, 6, dtype=torch.float64)
    queries = torch.randn(3, 5, dtype=torch.float64)

    errors = []
    for epsilon in (0.02, 0.01):
        scaled_key = epsilon * key
        scaled_queries = epsilon * queries
        logits = torch.einsum(
            "sh,bmh->bsm",
            scaled_queries,
            torch.einsum("hd,bmd->bmh", scaled_key, rows),
        ) / (key.shape[0] ** 0.5)
        exact = torch.einsum("bsm,bmd->bsd", torch.softmax(logits, -1), rows)
        approximation = PRECONDITIONER.qk_only_routing_nqf(
            rows,
            scaled_key,
            scaled_queries,
        )
        errors.append((exact - approximation).norm())

    assert errors[0] / errors[1] > 14.0


def test_full_and_qk_only_attention_match_nqf_gradient_orders() -> None:
    config = AUDIT.AuditConfig(
        batch_size=96,
        epsilon_count=9,
        epsilon_min_power=-3.5,
        epsilon_max_power=-0.8,
    )
    _, summary = AUDIT.run_audit(config)
    assert summary["all_checks_pass"]

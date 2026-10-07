#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-b963fcc
campaign_root=/root/MMNN/ICL-PDE/data/transformers/finite_basis_pde_minimal_v1
python_bin=/root/MMNN/.venv/bin/python
cohort_marker="$campaign_root/CAMPAIGN_COMPLETE"
audit_json="$campaign_root/functional_generalization_common_rotated_audit.json"
samples_csv="$campaign_root/functional_generalization_common_rotated_samples.csv"
causal_marker="$campaign_root/CAUSAL_FUNCTIONAL_AUDIT_COMPLETE.json"
assessment_json="$campaign_root/causal_icl_claim_assessment.json"

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-causal-icl-assessment

while ! "$python_bin" -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
assert payload["schema_version"] == 2
assert payload["status"] == "cohort_audits_complete"
assert payload["artifact_role"] == "integrity_and_descriptive_audits_not_claim_certificate"
assert payload["causal_icl_claim_status"] == "not_assessed"
assert payload["track"] == "manifest_union"
assert payload["expected_runs"] == 48
assert payload["validated_runs"] == 48
assert payload["expected_primal_runs"] == 0
assert payload["certified_primal_runs"] == 0
' "$cohort_marker" 2>/dev/null; do
  sleep 60
done

assessment_command=(
  "$python_bin"
  "$code_root/experiments/transformers/assess_finite_basis_causal_icl.py"
  --audit "$audit_json"
  --samples-csv "$samples_csv"
  --manifest "$campaign_root/MANIFEST_k4.json"
  --manifest "$campaign_root/MANIFEST_k8.json"
  --inventory-summary "$campaign_root/primal_exact_residual_stability_audit/summary.json"
  --cohort-marker "$cohort_marker"
  --causal-marker "$causal_marker"
  --output "$assessment_json"
)

if [[ -s "$audit_json" && -s "$samples_csv" && -s "$causal_marker" ]]; then
  if "${assessment_command[@]}"; then
    exit 0
  fi
fi

"$python_bin" \
  "$code_root/experiments/transformers/audit_finite_basis_functional_generalization.py" \
  --root "$campaign_root" \
  --run-dir "$campaign_root/k4_m0_4_query_attention_seed0" \
  --run-dir "$campaign_root/k4_m0_4_query_attention_seed1" \
  --run-dir "$campaign_root/k4_m0_4_query_attention_seed2" \
  --run-dir "$campaign_root/k4_m0_8_query_attention_seed0" \
  --run-dir "$campaign_root/k4_m0_8_query_attention_seed1" \
  --run-dir "$campaign_root/k4_m0_8_query_attention_seed2" \
  --run-dir "$campaign_root/k4_m0_16_query_attention_seed0" \
  --run-dir "$campaign_root/k4_m0_16_query_attention_seed1" \
  --run-dir "$campaign_root/k4_m0_16_query_attention_seed2" \
  --run-dir "$campaign_root/k8_m0_8_query_attention_seed0" \
  --run-dir "$campaign_root/k8_m0_8_query_attention_seed1" \
  --run-dir "$campaign_root/k8_m0_8_query_attention_seed2" \
  --run-dir "$campaign_root/k8_m0_16_query_attention_seed0" \
  --run-dir "$campaign_root/k8_m0_16_query_attention_seed1" \
  --run-dir "$campaign_root/k8_m0_16_query_attention_seed2" \
  --run-dir "$campaign_root/k8_m0_32_query_attention_seed0" \
  --run-dir "$campaign_root/k8_m0_32_query_attention_seed1" \
  --run-dir "$campaign_root/k8_m0_32_query_attention_seed2" \
  --device cuda \
  --batch-size 16 \
  --batches 16 \
  --prompt-lengths 128 \
  --max-prompt-length 128 \
  --audit-seed 314159265 \
  --task-law common_rotated \
  --completion-marker CAUSAL_FUNCTIONAL_AUDIT_COMPLETE.json

"${assessment_command[@]}"

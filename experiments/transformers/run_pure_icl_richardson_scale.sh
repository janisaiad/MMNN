#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/janis/STG3A/MMNN"
PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/experiments/transformers/pure_icl_parametric_operator_richardson_attention.py"
LOG="${ROOT}/data/transformers/runs_pure_icl_richardson_scale.log"

cd "${ROOT}/experiments/transformers"

run_one() {
  local name="$1"
  shift
  echo "" | tee -a "${LOG}"
  echo "=== START $(date -Is) ${name} ===" | tee -a "${LOG}"
  PYTHONUNBUFFERED=1 "${PY}" -u "${SCRIPT}" "$@" 2>&1 | tee -a "${LOG}"
  echo "=== DONE $(date -Is) ${name} ===" | tee -a "${LOG}"
}

echo "MASTER LOG $(date -Is)" | tee "${LOG}"

# Same geometry as runs_pure_icl_train_exact
run_one "rich_dual_baseline" \
  --mode train --solver dual_attention_richardson \
  --d 32 --K 8 --m 16 --R 32 \
  --z-depth 16 --heads 1 --d-head 64 \
  --learn-dictionary 1 --learn-probes 0 \
  --steps 30000 --batch-size 128 --eval-batch-size 512 \
  --log-every 250 --save-every 5000 --device cuda \
  --outdir runs_pure_icl_rich_dual_baseline

run_one "rich_dual_below" \
  --mode train --solver dual_attention_richardson \
  --d 32 --K 8 --m 8 --R 16 \
  --z-depth 16 --heads 1 --d-head 64 \
  --learn-dictionary 1 --learn-probes 0 \
  --steps 30000 --batch-size 128 --eval-batch-size 512 \
  --log-every 250 --save-every 5000 --device cuda \
  --outdir runs_pure_icl_rich_dual_below

run_one "rich_dual_highK" \
  --mode train --solver dual_attention_richardson \
  --d 32 --K 16 --m 16 --R 32 \
  --z-depth 16 --heads 1 --d-head 64 \
  --learn-dictionary 1 --learn-probes 0 \
  --steps 30000 --batch-size 64 --eval-batch-size 512 \
  --log-every 250 --save-every 5000 --device cuda \
  --outdir runs_pure_icl_rich_dual_highK

run_one "exact_reference" \
  --mode train --solver exact \
  --d 32 --K 8 --m 16 --R 32 \
  --learn-dictionary 1 --learn-probes 0 \
  --steps 30000 --batch-size 256 --eval-batch-size 512 \
  --log-every 250 --save-every 5000 --device cuda \
  --outdir runs_pure_icl_rich_exact_ref

echo "=== ALL DONE $(date -Is) ===" | tee -a "${LOG}"

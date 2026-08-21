"""Census Jacobian spectra of equilibria reached in the direct ODE audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_curl_census import field_jacobian
from experiments.spectral_self_attention.small_step_continuation import stack_models


def run(inputs: list[Path]) -> dict[str, object]:
    rows = []
    for path in inputs:
        payload = json.loads(path.read_text())
        beta0 = "beta0" in path.stem
        fixed = [record for record in payload["records"] if record["metrics"]["fixed"]]
        groups: dict[int, list[dict[str, object]]] = defaultdict(list)
        for record in fixed:
            groups[int(record["identity"]["n_tokens"])].append(record)
        for n_tokens, records in groups.items():
            models = stack_models(records)
            angles = np.asarray([record["final_angle"] for record in records])[:, None, :]
            jacobian = field_jacobian(angles, models)
            jacobian *= models["step_size"][:, None, None]
            for index, record in enumerate(records):
                eigenvalues = np.linalg.eigvals(jacobian[index])
                rows.append(
                    {
                        "file": path.name,
                        "source": (
                            f"type{int(record['family'])}_beta0"
                            if beta0
                            else f"type{int(record['family'])}"
                        ),
                        "family": int(record["family"]),
                        "label": str(record["label"]),
                        "n_tokens": n_tokens,
                        "subtype_code": int(record["identity"]["subtype_code"]),
                        "source_model_index": int(
                            record["identity"]["source_model_index"]
                        ),
                        "eigenvalues": [
                            {"real": float(value.real), "imag": float(value.imag)}
                            for value in eigenvalues
                        ],
                        "maximum_real": float(np.max(eigenvalues.real)),
                        "maximum_imaginary_absolute": float(
                            np.max(np.abs(eigenvalues.imag))
                        ),
                        "stable": bool(np.max(eigenvalues.real) < -1e-3),
                        "stable_spiral": bool(
                            np.max(eigenvalues.real) < -1e-3
                            and np.max(np.abs(eigenvalues.imag)) > 1e-3
                        ),
                    }
                )
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        key = f"{row['source']}_{row['label']}"
        entry = summary.setdefault(
            key, {"equilibria": 0, "stable": 0, "stable_spiral": 0}
        )
        entry["equilibria"] += 1
        entry["stable"] += int(row["stable"])
        entry["stable_spiral"] += int(row["stable_spiral"])
    return {"summary": summary, "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

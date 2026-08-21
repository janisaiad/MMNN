"""Refine every harvested p3/p4 candidate and certify primitive stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.periodic_orbit_audit import (
    block_from_record,
    refine_example,
)


def run(inputs: list[Path]) -> dict[str, object]:
    certificates = []
    for path in inputs:
        source = json.loads(path.read_text())
        family = int(source["family"])
        for label in ("p3", "p4"):
            period = int(label[1:])
            for record in source["records"][label]:
                block = block_from_record({"model": record["model"]}, family)
                certificate = refine_example(
                    block, np.asarray(record["angle"], dtype=float), period
                )
                certificates.append(
                    {
                        "family": family,
                        "n_tokens": int(record["n_tokens"]),
                        "subtype_code": int(record["subtype_code"]),
                        "label": label,
                        "source_model_index": int(record["source_model_index"]),
                        "screen_residual": record["screen_periodic_residual"],
                        "certificate": certificate,
                    }
                )
    summary: dict[str, dict[str, int]] = {}
    for record in certificates:
        key = f"type{record['family']}_{record['label']}"
        entry = summary.setdefault(
            key,
            {"candidates": 0, "primitive": 0, "stable": 0, "primitive_and_stable": 0},
        )
        entry["candidates"] += 1
        primitive = bool(record["certificate"]["primitive"])
        stable = bool(record["certificate"]["stable"])
        entry["primitive"] += int(primitive)
        entry["stable"] += int(stable)
        entry["primitive_and_stable"] += int(primitive and stable)
    return {"summary": summary, "certificates": certificates}


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

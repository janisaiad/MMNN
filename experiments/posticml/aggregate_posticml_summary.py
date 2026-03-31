#!/usr/bin/env python3
"""Collect key metrics from type_a / type_b / type_c result folders into SUMMARY.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
out = {"type_a": [], "type_b": {}, "type_c": []}

sa = RES / "type_a_synthetic_m1024"
if (sa / "summary_type_a.json").exists():
    with open(sa / "summary_type_a.json") as f:
        s = json.load(f)
    out["type_a_meta"] = {k: s[k] for k in ("M_width", "sqrt_M", "note") if k in s}
    for run in s.get("runs", []):
        name = run.get("name")
        if not name:
            continue
        lj = sa / name / "losses.json"
        if lj.exists():
            with open(lj) as f:
                lj_data = json.load(f)
            out["type_a"].append(
                {
                    "name": name,
                    "hidden_rank": lj_data.get("config", {}).get("hidden_rank"),
                    "final_test_mse": lj_data.get("final_test_error"),
                    "epochs": lj_data.get("epochs_run"),
                }
            )

sb = RES / "type_b_cifar"
out["type_b"] = {}
for p in sorted(sb.glob("cifar10_*.json")):
    with open(p) as f:
        d = json.load(f)
    out["type_b"][p.name] = {
        "best_test_acc": d.get("best_test_acc"),
        "final_test_acc": d.get("final_test_acc"),
        "epochs": d.get("epochs"),
    }

sc = RES / "type_c_highd_spikes"
for p in sorted(sc.glob("**/results.json")):
    with open(p) as f:
        d = json.load(f)
    out["type_c"].append({"dir": str(p.parent.name), **{k: d.get(k) for k in (
        "d", "n_train", "hidden_rank", "final_test_mse_raw", "final_test_mse_z", "test_nmse_var", "epochs_run",
    )}})

with open(ROOT / "SUMMARY.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))

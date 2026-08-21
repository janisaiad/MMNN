"""Merge token-count shards produced by the large attractor census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_count_dict(destination: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        destination[key] = destination.get(key, 0) + int(value)


def merge(paths: list[Path]) -> dict[str, object]:
    shards = [json.loads(path.read_text()) for path in paths]
    first = shards[0]
    output = {
        "family": first["family"],
        "family_name": first["family_name"],
        "settings": dict(first["settings"]),
        "totals": {},
        "model_incidence": {},
        "by_tokens": {},
        "grouped": {},
        "examples": {},
    }
    output["settings"]["token_counts"] = sorted(
        int(tokens) for shard in shards for tokens in shard["by_tokens"]
    )
    for shard in shards:
        add_count_dict(output["totals"], shard["totals"])
        add_count_dict(output["model_incidence"], shard["model_incidence"])
        output["by_tokens"].update(shard["by_tokens"])
        for group_name, values in shard["grouped"].items():
            destination_values = output["grouped"].setdefault(group_name, {})
            for label, counts in values.items():
                add_count_dict(destination_values.setdefault(label, {}), counts)
        for name, example in shard["examples"].items():
            output["examples"].setdefault(name, example)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = merge(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"family": result["family_name"], "totals": result["totals"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

LIBERO_SOURCES = {"LIBERO-Spatial", "LIBERO-Object", "LIBERO-Goal", "LIBERO-10"}
GRID_KEYS = {
    "grid_flow",
    "grid_flow_path",
    "grid_flow_npz",
    "grid_flow_meta",
    "grid_flow_meta_path",
    "grid_teacher",
    "grid_teacher_path",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = {k: v for k, v in row.items() if k != "_source_line_no"}
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def is_libero(row: dict[str, Any]) -> bool:
    return row.get("source_dataset") in LIBERO_SOURCES


def without_grid(row: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(row)
    loss_mask = dict(out.get("loss_mask") or {})
    loss_mask["gridflow"] = 0.0
    out["loss_mask"] = loss_mask
    for key in GRID_KEYS:
        out.pop(key, None)
    # Keep canonical trajectory files untouched. The no-grid loader switch
    # prevents filesystem dependency on flow/grid_flow.*.
    return out


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = Counter()
    family = Counter()
    for row in rows:
        source[str(row.get("source_dataset"))] += 1
        family[str(row.get("dataset_family"))] += 1
    return {
        "row_count": len(rows),
        "source_distribution": dict(sorted(source.items())),
        "family_distribution": dict(sorted(family.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser("Build IF-WAM manifest ablation splits.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/2024233240/if-wam_incoming/ifwam_data/manifests/train_mixed_grid_rgb_plus_libero10.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-libero-rows", type=int, default=1711)
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir or input_path.parent
    rows = read_jsonl(input_path)
    libero_rows = [row for row in rows if is_libero(row)]

    outputs = {
        "train_libero_all_from_mixed_grid_rgb.jsonl": libero_rows,
        "train_libero_all_from_mixed_nogrid.jsonl": [without_grid(row) for row in libero_rows],
        "train_mixed_grid_rgb_plus_libero10_nogrid.jsonl": [without_grid(row) for row in rows],
    }

    if len(libero_rows) != args.expected_libero_rows:
        raise RuntimeError(
            f"Expected LIBERO-only rows={args.expected_libero_rows}, got {len(libero_rows)} from {input_path}"
        )

    report = {"input": str(input_path), "outputs": {}}
    for name, out_rows in outputs.items():
        path = output_dir / name
        write_jsonl(path, out_rows)
        report["outputs"][name] = {"path": str(path), **distribution(out_rows)}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

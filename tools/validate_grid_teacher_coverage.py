#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser("Validate IF-WAM grid teacher pair coverage for a training manifest.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--max-missing-report", type=int, default=20)
    args = ap.parse_args()

    total_rows = 0
    total_pairs = 0
    missing_rows = 0
    missing_pairs = 0
    missing_files = 0
    source_counts: Counter[str] = Counter()
    missing_examples = []
    cache: dict[str, set[tuple[int, int]] | None] = {}

    for line_no, row in iter_jsonl(args.manifest):
        total_rows += 1
        source_counts[str(row.get("source_dataset") or "unknown")] += 1
        starts = row.get("flow_start_indices") or []
        ends = row.get("flow_end_indices") or []
        required = [(int(s), int(e)) for s, e in zip(starts, ends)]
        total_pairs += len(required)
        if not required:
            continue

        traj_dir = str(row["traj_dir"])
        pair_set = cache.get(traj_dir)
        if traj_dir not in cache:
            path = Path(traj_dir) / "flow" / "grid_flow.npz"
            if not path.exists():
                pair_set = None
                missing_files += 1
            else:
                with np.load(path, allow_pickle=False) as data:
                    s_arr = np.asarray(data.get("start_frame_idx", []), dtype=np.int64)
                    e_arr = np.asarray(data.get("end_frame_idx", []), dtype=np.int64)
                pair_set = {(int(s), int(e)) for s, e in zip(s_arr, e_arr)}
            cache[traj_dir] = pair_set

        row_missing = []
        if pair_set is None:
            row_missing = required
        else:
            row_missing = [pair for pair in required if pair not in pair_set]
        if row_missing:
            missing_rows += 1
            missing_pairs += len(row_missing)
            if len(missing_examples) < args.max_missing_report:
                missing_examples.append(
                    {
                        "line_no": line_no,
                        "traj_dir": traj_dir,
                        "required": required,
                        "missing": row_missing,
                        "source_dataset": row.get("source_dataset"),
                    }
                )

    summary = {
        "manifest": str(args.manifest),
        "total_rows": total_rows,
        "total_pairs": total_pairs,
        "unique_traj": len(cache),
        "source_counts": dict(source_counts),
        "missing_files": missing_files,
        "missing_rows": missing_rows,
        "missing_pairs": missing_pairs,
        "coverage_ok": missing_rows == 0 and missing_pairs == 0 and missing_files == 0,
        "missing_examples": missing_examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["coverage_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

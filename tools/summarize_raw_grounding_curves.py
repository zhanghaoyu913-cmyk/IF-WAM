#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "global_step",
    "unweighted_action_loss",
    "weighted_action_loss",
    "action_prediction_norm",
    "action_target_norm",
    "grad_norm",
    "lr",
    "total_loss",
]


def load_nearest(metrics_path: Path, requested_steps: list[int]) -> list[dict]:
    rows = []
    if not metrics_path.exists():
        return rows
    by_step = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "global_step" in row:
                by_step[int(row["global_step"])] = row
    if not by_step:
        return rows
    available = sorted(by_step)
    for step in requested_steps:
        nearest = min(available, key=lambda s: abs(s - step))
        row = dict(by_step[nearest])
        row["requested_step"] = step
        row["matched_step"] = nearest
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR containing train_metrics.jsonl")
    ap.add_argument("--output-dir", type=Path, default=Path("reports/raw_grounding_curves"))
    ap.add_argument("--steps", default="0,500,1000,2000,4000,8679")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_steps = [int(x) for x in args.steps.split(",") if x.strip()]
    all_rows = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be NAME=RUN_DIR, got {spec!r}")
        name, run_dir = spec.split("=", 1)
        run_dir = Path(run_dir)
        metrics_path = run_dir / "train_metrics.jsonl"
        rows = load_nearest(metrics_path, requested_steps)
        for row in rows:
            out = {"run": name, "run_dir": str(run_dir), "metrics_path": str(metrics_path)}
            for field in ["requested_step", "matched_step", *FIELDS]:
                out[field] = row.get(field, "")
            all_rows.append(out)

    csv_path = args.output_dir / "grounding_curve_metrics.csv"
    fieldnames = ["run", "requested_step", "matched_step", *FIELDS, "run_dir", "metrics_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    md_path = args.output_dir / "grounding_curve_summary.md"
    lines = [
        "# Raw Grounding Curve Summary",
        "",
        "This report is generated from `train_metrics.jsonl`. Missing metrics indicate the run predates the corresponding logging field or has not completed that step.",
        "",
        "| run | requested step | matched step | action loss | pred norm | target norm | grad norm | lr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in all_rows:
        def fmt(v):
            if v == "" or v is None:
                return "NA"
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    fmt(row["requested_step"]),
                    fmt(row["matched_step"]),
                    fmt(row.get("unweighted_action_loss", "")),
                    fmt(row.get("action_prediction_norm", "")),
                    fmt(row.get("action_target_norm", "")),
                    fmt(row.get("grad_norm", "")),
                    fmt(row.get("lr", "")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "rows": len(all_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

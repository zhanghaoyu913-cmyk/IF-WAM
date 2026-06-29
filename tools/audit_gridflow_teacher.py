#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ifwam.gridflow_utils import adaptive_moving_mask_numpy


META_KEYS = (
    "grid_assignment",
    "window_rule",
    "scale_mode",
    "direction_reliable",
    "magnitude_reliable",
    "flow_coordinate_frame",
    "flow_vector_definition",
    "projection_coordinate_frame",
    "camera_extrinsic_convention",
    "coord_mode",
    "flow_camera",
)


def _percentiles(values):
    if not values:
        return {f"p{p}": None for p in (0, 10, 25, 50, 75, 90, 100)}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.nanpercentile(arr, p)) for p in (0, 10, 25, 50, 75, 90, 100)}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_heatmap(path: Path, arr: np.ndarray, title: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-visualizations", type=int, default=0)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if args.source_name and row.get("source_dataset") != args.source_name:
                    continue
                rows.append(row)
                if args.max_samples and len(rows) >= args.max_samples:
                    break

    stats = {
        "samples": len(rows),
        "teacher_exists": 0,
        "window_matches": 0,
        "nan_inf_files": 0,
        "all_zero_windows": 0,
        "meta_unique_values": {key: Counter() for key in META_KEYS},
        "start_end_pairs": Counter(),
    }
    by_source = defaultdict(lambda: Counter(samples=0, teacher_exists=0, window_matches=0))
    by_task = defaultdict(lambda: Counter(samples=0, teacher_exists=0, window_matches=0))
    valid_ratios = []
    moving_ratios = []
    qualities = []
    norm_all = []
    norm_moving = []
    track_counts = []
    thresholds = []
    resultant_lengths = []
    angular_dispersions = []
    quality_track_pairs = []
    quality_stability_pairs = []
    axis_hist = Counter()
    visualized = 0

    for row in rows:
        source = row.get("source_dataset") or "unknown"
        task = row.get("task_label") or "unknown"
        by_source[source]["samples"] += 1
        by_task[task]["samples"] += 1
        traj_dir = Path(row["traj_dir"])
        npz_value = row.get("grid_flow_path") or row.get("grid_flow_npz")
        npz_path = Path(npz_value) if npz_value else traj_dir / "flow" / "grid_flow.npz"
        if not npz_path.is_absolute():
            npz_path = traj_dir / npz_path
        meta_value = row.get("grid_flow_meta_path") or row.get("grid_flow_meta")
        meta_path = Path(meta_value) if meta_value else npz_path.with_name("grid_flow_meta.json")
        if not meta_path.is_absolute():
            meta_path = traj_dir / meta_path
        if not npz_path.exists():
            continue
        stats["teacher_exists"] += 1
        by_source[source]["teacher_exists"] += 1
        by_task[task]["teacher_exists"] += 1
        meta = _read_json(meta_path)
        for key in META_KEYS:
            value = meta.get(key, "<missing>")
            if isinstance(value, dict):
                value = json.dumps(value, sort_keys=True)
            stats["meta_unique_values"][key][str(value)] += 1
        with np.load(npz_path, allow_pickle=False) as data:
            flow = np.asarray(data["grid_flow_vectors"], dtype=np.float32)
            valid = np.asarray(data["grid_valid_mask"], dtype=np.float32)
            quality = np.asarray(data.get("grid_quality", np.ones(flow.shape[0])), dtype=np.float32)
            starts = np.asarray(data.get("start_frame_idx", []), dtype=np.int64)
            ends = np.asarray(data.get("end_frame_idx", []), dtype=np.int64)
            counts = np.asarray(data.get("cell_track_counts", np.zeros(valid.shape)), dtype=np.float32)
        requested = set()
        for s, e in zip(row.get("flow_start_indices", []), row.get("flow_end_indices", [])):
            requested.add((int(s), int(e)))
        existing = {(int(s), int(e)) for s, e in zip(starts, ends)}
        if not requested or requested.issubset(existing):
            stats["window_matches"] += 1
            by_source[source]["window_matches"] += 1
            by_task[task]["window_matches"] += 1
        for pair in existing:
            stats["start_end_pairs"][str(pair)] += 1
        if not np.isfinite(flow).all() or not np.isfinite(valid).all() or not np.isfinite(quality).all():
            stats["nan_inf_files"] += 1
        norms = np.linalg.norm(flow, axis=-1)
        valid_bool = valid > 0
        valid_ratios.extend(valid_bool.reshape(valid.shape[0], -1).mean(axis=1).tolist())
        qualities.extend(quality.tolist())
        norm_all.extend(norms[valid_bool].tolist())
        moving, threshold = adaptive_moving_mask_numpy(flow, valid, min_motion_threshold=1.0e-3, mad_scale=3.0)
        thresholds.extend(threshold.tolist())
        moving_ratios.extend(moving.reshape(moving.shape[0], -1).mean(axis=1).tolist())
        norm_moving.extend(norms[moving].tolist())
        track_counts.extend(counts[valid_bool].tolist())
        if valid_bool.any():
            quality_track_pairs.extend([(float(q), float(c)) for q, c in zip(np.repeat(quality, valid.shape[1] * valid.shape[2])[valid_bool.reshape(-1)], counts.reshape(-1)[valid_bool.reshape(-1)])])
        for wi in range(flow.shape[0]):
            mask = moving[wi]
            if not mask.any():
                continue
            vec = flow[wi][mask]
            unit = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-6)
            mean_vec = unit.mean(axis=0)
            resultant = float(np.linalg.norm(mean_vec))
            resultant_lengths.append(resultant)
            angular_dispersions.append(float(np.rad2deg(np.arccos(np.clip(resultant, -1.0, 1.0)))))
            quality_stability_pairs.append((float(quality[wi]), resultant))
            dominant = np.argmax(np.abs(unit), axis=-1)
            signs = np.sign(unit[np.arange(unit.shape[0]), dominant])
            for ax, sign in zip(dominant, signs):
                axis_hist[f"{'xyz'[int(ax)]}{'+' if sign >= 0 else '-'}"] += 1
        stats["all_zero_windows"] += int(((np.abs(flow).sum(axis=(-1, -2, -3)) == 0) | (~valid_bool.reshape(valid.shape[0], -1).any(axis=1))).sum())
        if visualized < args.num_visualizations:
            wi = 0
            prefix = viz_dir / f"sample_{visualized:04d}"
            _write_heatmap(prefix.with_name(prefix.name + "_valid.png"), valid[wi], "valid mask")
            _write_heatmap(prefix.with_name(prefix.name + "_norm.png"), norms[wi], "flow norm")
            _write_heatmap(prefix.with_name(prefix.name + "_x.png"), flow[wi, :, :, 0], "x direction")
            _write_heatmap(prefix.with_name(prefix.name + "_y.png"), flow[wi, :, :, 1], "y direction")
            _write_heatmap(prefix.with_name(prefix.name + "_z.png"), flow[wi, :, :, 2], "z direction")
            _write_heatmap(prefix.with_name(prefix.name + "_counts.png"), counts[wi], "cell track counts")
            _write_heatmap(prefix.with_name(prefix.name + "_moving.png"), moving[wi].astype(np.float32), "moving mask")
            visualized += 1

    summary = {
        "samples": stats["samples"],
        "teacher_file_exists_rate": stats["teacher_exists"] / max(1, stats["samples"]),
        "window_exact_match_rate": stats["window_matches"] / max(1, stats["samples"]),
        "valid_cell_ratio": _percentiles(valid_ratios),
        "moving_cell_ratio": _percentiles(moving_ratios),
        "grid_quality": _percentiles(qualities),
        "flow_norm_all_valid": _percentiles(norm_all),
        "flow_norm_moving": _percentiles(norm_moving),
        "cell_track_counts": _percentiles(track_counts),
        "moving_threshold": _percentiles(thresholds),
        "unit_vector_resultant_length": _percentiles(resultant_lengths),
        "point_flow_direction_angular_dispersion_deg": _percentiles(angular_dispersions),
        "quality_track_count_corr": None,
        "quality_direction_stability_corr": None,
        "world_axis_direction_histogram": dict(axis_hist),
        "nan_inf_files": stats["nan_inf_files"],
        "all_zero_windows": stats["all_zero_windows"],
        "start_end_pair_distribution": dict(stats["start_end_pairs"].most_common()),
        "meta_unique_values": {k: dict(v) for k, v in stats["meta_unique_values"].items()},
    }
    if len(quality_track_pairs) > 1:
        arr = np.asarray(quality_track_pairs, dtype=np.float64)
        summary["quality_track_count_corr"] = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    if len(quality_stability_pairs) > 1:
        arr = np.asarray(quality_stability_pairs, dtype=np.float64)
        summary["quality_direction_stability_corr"] = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for name, table in (("audit_by_source.csv", by_source), ("audit_by_task.csv", by_task)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["key", "samples", "teacher_exists", "window_matches", "teacher_exists_rate", "window_match_rate"])
            writer.writeheader()
            for key, c in sorted(table.items()):
                samples = max(1, c["samples"])
                writer.writerow({
                    "key": key,
                    "samples": c["samples"],
                    "teacher_exists": c["teacher_exists"],
                    "window_matches": c["window_matches"],
                    "teacher_exists_rate": c["teacher_exists"] / samples,
                    "window_match_rate": c["window_matches"] / samples,
                })
    for src, dst in (("audit_by_source.csv", "by_source.csv"), ("audit_by_task.csv", "by_task.csv")):
        (out_dir / dst).write_text((out_dir / src).read_text(encoding="utf-8"), encoding="utf-8")

    report = [
        "# Grid-flow Teacher Audit",
        "",
        f"- samples: {summary['samples']}",
        f"- teacher file exists rate: {summary['teacher_file_exists_rate']:.4f}",
        f"- requested-window exact match rate: {summary['window_exact_match_rate']:.4f}",
        f"- valid cell ratio p50: {summary['valid_cell_ratio']['p50']}",
        f"- moving cell ratio p50: {summary['moving_cell_ratio']['p50']}",
        f"- grid quality p50: {summary['grid_quality']['p50']}",
        f"- flow norm moving p50: {summary['flow_norm_moving']['p50']}",
        "",
        "## Coordinate Frame",
        "",
    ]
    coord_values = summary["meta_unique_values"].get("flow_coordinate_frame", {})
    if not coord_values or list(coord_values) == ["<missing>"]:
        report.append("HIGH RISK: meta does not explicitly record `flow_coordinate_frame`.")
    else:
        report.append(f"`flow_coordinate_frame`: {coord_values}")
    report.append("")
    report.append("Builder code computes `grid_flow_vectors = p_end - p_start` from TraceForge `coords`, then projects `p_start` with the start-frame extrinsic for grid assignment. Existing data without the new meta fields should be treated as high-risk until audited.")
    (out_dir / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out_dir / "coordinate_frame_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    old_vs = out_dir / "old_vs_regenerated_teacher.csv"
    if not old_vs.exists():
        old_vs.write_text("status,reason\nnot_run,regeneration requires original TraceForge records and is intentionally not inferred from filenames\n", encoding="utf-8")


if __name__ == "__main__":
    main()

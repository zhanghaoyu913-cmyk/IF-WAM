#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from ifwam.data.layout_batch_sampler import LayoutHomogeneousBatchSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mixed_gridfm_one_way_direction_presence")
    parser.add_argument("--num-batches", type=int, default=5000)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-dir", default="reports/sampler_exposure")
    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(Path.cwd() / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.config}", "num_workers=0"])
    ds = instantiate(cfg.data.train)
    sampler = LayoutHomogeneousBatchSampler(
        dataset=ds,
        batch_size=int(cfg.batch_size),
        seed=int(cfg.seed),
        drop_last=bool(getattr(ds, "sampler_drop_last", True)),
        group_sampling=str(getattr(ds, "sampler_group_sampling", "proportional_to_num_rows")),
        shuffle=bool(getattr(ds, "sampler_shuffle", True)),
    )
    batches = list(sampler)
    rank_counts = []
    rank_teacher = []
    rank_grid_valid = []
    for rank in range(max(1, args.world_size)):
        counts = Counter()
        teacher = 0
        grid_valid = 0
        seen = 0
        for batch in batches[rank::max(1, args.world_size)]:
            if seen >= args.num_batches:
                break
            for idx in batch:
                counts[str(ds.source_key(idx) if hasattr(ds, "source_key") else "unknown")] += 1
                row = ds.rows[idx]
                if (Path(row["traj_dir"]) / "flow" / "grid_flow.npz").exists():
                    teacher += 1
                    grid_valid += 1
            seen += 1
        rank_counts.append(dict(counts))
        rank_teacher.append(teacher)
        rank_grid_valid.append(grid_valid)
    total = Counter()
    for c in rank_counts:
        total.update(c)
    out = {
        "config": args.config,
        "intended_source_weights": sampler.effective_source_weights(),
        "observed_source_counts": dict(total),
        "counts_by_rank": rank_counts,
        "libero_samples_seen": sum(v for k, v in total.items() if k.startswith("LIBERO")),
        "grid_valid_samples_seen": sum(rank_grid_valid),
        "teacher_covered_samples_seen": sum(rank_teacher),
        "world_size": args.world_size,
        "num_batches_per_rank_requested": args.num_batches,
    }
    total_n = sum(total.values()) or 1
    intended = sampler.effective_source_weights()
    if intended:
        weight_sum = sum(float(v) for v in intended.values()) or 1.0
        out["expected_vs_observed_deviation"] = {
            k: (total.get(k, 0) / total_n) - (float(v) / weight_sum)
            for k, v in intended.items()
        }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sampler_exposure.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Source Sampler Exposure", "", f"- config: {args.config}", f"- world_size: {args.world_size}", f"- observed_source_counts: {dict(total)}", f"- intended_source_weights: {sampler.effective_source_weights()}", f"- LIBERO samples seen: {out['libero_samples_seen']}", f"- grid-valid samples seen: {out['grid_valid_samples_seen']}", f"- teacher-covered samples seen: {out['teacher_covered_samples_seen']}"]
    (out_dir / "sampler_exposure.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

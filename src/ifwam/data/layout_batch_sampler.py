from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Iterator

import torch
from torch.utils.data import Sampler


class LayoutHomogeneousBatchSampler(Sampler[list[int]]):
    """Yield batches whose already-built samples share camera layout and action schema.

    Camera concatenation remains inside each dataset sample. This sampler never combines
    cameras from different trajectories; it only groups complete samples for tensor stacking.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = True,
        group_sampling: str = "proportional_to_num_rows",
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.group_sampling = str(group_sampling)
        self.shuffle = bool(shuffle)
        self.source_weights = {str(k): float(v) for k, v in getattr(dataset, "sampler_source_weights", {}).items()}
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        groups = defaultdict(list)
        for idx in range(len(dataset)):
            key = dataset.batch_group_key(idx)
            if self.group_sampling == "source_weighted" and hasattr(dataset, "source_key"):
                key = (key, dataset.source_key(idx))
            groups[key].append(idx)
        self.groups = dict(groups)
        if not self.groups:
            raise ValueError("LayoutHomogeneousBatchSampler requires a non-empty dataset.")
        if self.group_sampling not in {"proportional_to_num_rows", "uniform_by_group", "source_weighted"}:
            raise ValueError(
                "group_sampling must be 'proportional_to_num_rows', 'uniform_by_group', or 'source_weighted', "
                f"got {self.group_sampling!r}"
            )

    def group_counts(self) -> dict[str, int]:
        return {repr(key): len(indices) for key, indices in self.groups.items()}

    def effective_source_weights(self) -> dict[str, float]:
        return dict(sorted(self.source_weights.items()))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch + self.epoch_offset)
        grouped_batches = []
        for key, indices in self.groups.items():
            if self.shuffle:
                order = torch.randperm(len(indices), generator=generator).tolist()
            else:
                order = list(range(len(indices)))
            shuffled = [indices[i] for i in order]
            group_batches = []
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    group_batches.append(batch)
            if group_batches:
                grouped_batches.append((key, group_batches))

        if self.group_sampling == "uniform_by_group":
            max_len = max((len(group_batches) for _, group_batches in grouped_batches), default=0)
            batches = []
            for offset in range(max_len):
                for _, group_batches in grouped_batches:
                    if offset < len(group_batches):
                        batches.append(group_batches[offset])
        elif self.group_sampling == "source_weighted" and self.source_weights:
            batches_by_source = defaultdict(list)
            for key, group_batches in grouped_batches:
                indices = self.groups[key]
                source = str(self.dataset.source_key(indices[0])) if hasattr(self.dataset, "source_key") and indices else "unknown"
                weight = max(float(self.source_weights.get(source, self.source_weights.get("default", 1.0))), 0.0)
                if weight > 0:
                    batches_by_source[source].extend(group_batches)
            if not batches_by_source:
                batches = []
            else:
                max_base = max(len(v) / max(float(self.source_weights.get(k, self.source_weights.get("default", 1.0))), 1e-12) for k, v in batches_by_source.items())
                batches = []
                for source, source_batches in sorted(batches_by_source.items()):
                    weight = max(float(self.source_weights.get(source, self.source_weights.get("default", 1.0))), 0.0)
                    target = max(int(round(max_base * weight)), 1)
                    for i in range(target):
                        batches.append(source_batches[i % len(source_batches)])
        else:
            batches = [batch for _, group_batches in grouped_batches for batch in group_batches]

        if batches and self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]
        if self.epoch == 0 and self.resume_batch_offset > 0:
            batches = batches[self.resume_batch_offset :]
        return iter(batches)

    def __len__(self) -> int:
        def _group_batch_count(indices):
            return len(indices) // self.batch_size if self.drop_last else ceil(len(indices) / self.batch_size)
        if self.group_sampling == "source_weighted" and self.source_weights:
            source_batch_counts = defaultdict(int)
            for indices in self.groups.values():
                source = str(self.dataset.source_key(indices[0])) if hasattr(self.dataset, "source_key") and indices else "unknown"
                weight = max(float(self.source_weights.get(source, self.source_weights.get("default", 1.0))), 0.0)
                if weight > 0:
                    source_batch_counts[source] += _group_batch_count(indices)
            if not source_batch_counts:
                return 0
            max_base = max(count / max(float(self.source_weights.get(source, self.source_weights.get("default", 1.0))), 1e-12) for source, count in source_batch_counts.items())
            return sum(max(int(round(max_base * max(float(self.source_weights.get(source, self.source_weights.get("default", 1.0))), 0.0))), 1) for source in source_batch_counts)
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.groups.values())
        return sum(ceil(len(indices) / self.batch_size) for indices in self.groups.values())

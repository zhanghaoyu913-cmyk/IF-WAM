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

    def __init__(self, dataset, batch_size: int, seed: int = 42, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        groups = defaultdict(list)
        for idx in range(len(dataset)):
            groups[dataset.batch_group_key(idx)].append(idx)
        self.groups = dict(groups)
        if not self.groups:
            raise ValueError("LayoutHomogeneousBatchSampler requires a non-empty dataset.")

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
        batches = []
        for indices in self.groups.values():
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in order]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if batches:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]
        if self.epoch == 0 and self.resume_batch_offset > 0:
            batches = batches[self.resume_batch_offset :]
        return iter(batches)

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.groups.values())
        return sum(ceil(len(indices) / self.batch_size) for indices in self.groups.values())

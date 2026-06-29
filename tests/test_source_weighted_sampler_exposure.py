from collections import Counter

from ifwam.data.layout_batch_sampler import LayoutHomogeneousBatchSampler


class _FakeDataset:
    layout_homogeneous_batches = True
    sampler_source_weights = {"libero": 2.0, "robot": 1.0}

    def __init__(self):
        self.sources = ["libero"] * 8 + ["robot"] * 8

    def __len__(self):
        return len(self.sources)

    def batch_group_key(self, idx):
        return ("layout", self.sources[idx])

    def source_key(self, idx):
        return self.sources[idx]


def test_source_weighted_sampler_exposure():
    ds = _FakeDataset()
    sampler = LayoutHomogeneousBatchSampler(ds, batch_size=2, seed=0, shuffle=False, group_sampling="source_weighted")
    counts = Counter()
    for batch in sampler:
        for idx in batch:
            counts[ds.source_key(idx)] += 1
    assert counts["libero"] == 16
    assert counts["robot"] == 8
    assert sampler.effective_source_weights() == {"libero": 2.0, "robot": 1.0}

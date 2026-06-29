from collections import Counter

from ifwam.data.layout_batch_sampler import LayoutHomogeneousBatchSampler


class _FakeDataset:
    sampler_source_weights = {"libero": 3.0, "robot": 1.0}

    def __init__(self):
        self.sources = ["libero"] * 12 + ["robot"] * 12

    def __len__(self):
        return len(self.sources)

    def batch_group_key(self, idx):
        return ("layout", self.sources[idx])

    def source_key(self, idx):
        return self.sources[idx]


def test_source_weighted_sampler_ddp_rank_slicing_exposure():
    ds = _FakeDataset()
    sampler = LayoutHomogeneousBatchSampler(ds, batch_size=2, seed=0, shuffle=False, group_sampling="source_weighted")
    batches = list(sampler)
    rank_counts = []
    for rank in range(2):
        counts = Counter()
        for batch in batches[rank::2]:
            for idx in batch:
                counts[ds.source_key(idx)] += 1
        rank_counts.append(counts)
    total = rank_counts[0] + rank_counts[1]
    assert total["libero"] == 36
    assert total["robot"] == 12
    assert abs(rank_counts[0]["libero"] - rank_counts[1]["libero"]) <= 2

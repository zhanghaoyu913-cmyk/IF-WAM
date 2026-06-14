import json
from pathlib import Path

from ifwam.data import IFWAMManifestDataset, LayoutHomogeneousBatchSampler


def test_layout_sampler_keeps_complete_samples_grouped(tmp_path: Path):
    manifest = tmp_path / "train.jsonl"
    rows = [
        {"traj_dir": "/a", "camera_names": ["main", "wrist"], "action_type": "eef_delta_gripper", "action_dim": 7},
        {"traj_dir": "/b", "camera_names": ["main", "wrist"], "action_type": "eef_delta_gripper", "action_dim": 7},
        {"traj_dir": "/c", "camera_names": ["main"], "action_type": "none", "action_dim": 7},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ds = IFWAMManifestDataset(str(manifest), image_size=(32, 32), text_embedding_cache_dir=None)
    sampler = LayoutHomogeneousBatchSampler(ds, batch_size=2, seed=0)
    batches = list(sampler)
    for batch in batches:
        keys = {ds.batch_group_key(idx) for idx in batch}
        assert len(keys) == 1
    assert any(set(batch) == {0, 1} for batch in batches)

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ifwam.data import IFWAMManifestDataset


def test_manifest_dataset_fake_traj(tmp_path: Path):
    traj = tmp_path / "task" / "traj_000000"
    (traj / "frames" / "main").mkdir(parents=True)
    (traj / "flow").mkdir()
    for i in range(9):
        Image.new("RGB", (32, 32), (i, 0, 0)).save(traj / "frames" / "main" / f"{i:06d}.jpg")
        Image.new("RGB", (32, 32), (i, 0, 0)).save(traj / f"rgb_{i}.png")
    (traj / "language.txt").write_text("pick up cup")
    (traj / "semantics.json").write_text(json.dumps({"language": "pick up cup"}))
    (traj / "meta.json").write_text(json.dumps({"has_action": False}))
    np.savez_compressed(traj / "flow" / "teacher_flow.npz", flow_vectors=np.zeros((8,3,3), dtype=np.float32), valid_mask=np.ones((8,3), dtype=np.float32), quality=np.ones((8,), dtype=np.float32))
    np.savez_compressed(traj / "flow" / "grid_flow.npz", grid_flow_vectors=np.zeros((8,8,8,3), dtype=np.float32), grid_valid_mask=np.ones((8,8,8), dtype=np.float32), grid_quality=np.ones((8,), dtype=np.float32))
    manifest = tmp_path / "train.jsonl"
    row = {"traj_dir": str(traj), "dataset_family": "human", "source_dataset": "fake", "task_label": "task", "camera_names": ["main"], "frame_indices": list(range(9)), "action_start": None, "action_end": None, "loss_mask": {"video": 1, "action": 0, "vflow": 1, "aflow": 0, "mag": 0, "gridflow": 1, "progress": 0}}
    manifest.write_text(json.dumps(row) + "\n")
    ds = IFWAMManifestDataset(str(manifest), image_size=(32, 32), action_dim=7)
    item = ds[0]
    assert item["video"].shape == (3, 9, 32, 32)
    assert item["action"].shape == (32, 7)
    assert item["action_mask"].item() == 0.0
    assert item["iflow_teacher"].shape == (8, 3, 3)
    assert item["iflow_valid"].shape == (8, 3)
    assert item["grid_flow_teacher"].shape == (8, 8, 8, 3)
    assert item["grid_flow_valid"].shape == (8, 8, 8)
    assert item["loss_mask"]["action"] == 0.0
    assert item["loss_mask"]["gridflow"] == 1.0

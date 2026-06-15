from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

LOSS_KEYS = ("video", "action", "vflow", "aflow", "gridflow", "mag", "progress")
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_rgb(path: Path, size: tuple[int, int]) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, list(size), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    tensor = TF.to_tensor(image)
    return tensor.mul(2.0).sub(1.0)


class IFWAMManifestDataset(Dataset):
    """Dataset for IF-WAM training manifests.

    It returns schema-native keys and Fast-WAM-compatible keys (`prompt`,
    `image_is_pad`, `action_is_pad`) so copied Fast-WAM code can consume it.
    """

    def __init__(
        self,
        manifest_path: str,
        image_size: Sequence[int] | tuple[int, int],
        num_video_frames: int = 9,
        num_action_steps: int = 32,
        num_flow_windows: int = 8,
        camera_mode: str = "concat",
        action_dim: int = 7,
        state_dim: int = 0,
        proprio_dim: int = 0,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
        context_dim: int = 4096,
        grid_size: Sequence[int] | tuple[int, int] = (8, 8),
        prompt_template: str = DEFAULT_PROMPT,
        layout_homogeneous_batches: bool = True,
        enable_gridflow_teacher: bool = True,
        sampler_group_sampling: str = "proportional_to_num_rows",
        sampler_shuffle: bool = True,
        sampler_drop_last: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.num_video_frames = int(num_video_frames)
        self.num_action_steps = int(num_action_steps)
        self.num_flow_windows = int(num_flow_windows)
        self.camera_mode = str(camera_mode)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.proprio_dim = int(proprio_dim)
        self.text_embedding_cache_dir = None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir)
        self.context_len = int(context_len)
        self.context_dim = int(context_dim)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.prompt_template = str(prompt_template)
        self.layout_homogeneous_batches = bool(layout_homogeneous_batches)
        self.enable_gridflow_teacher = bool(enable_gridflow_teacher)
        self.sampler_group_sampling = str(sampler_group_sampling)
        self.sampler_shuffle = bool(sampler_shuffle)
        self.sampler_drop_last = bool(sampler_drop_last)
        from .collate import collate_ifwam_batch
        self.collate_fn = collate_ifwam_batch
        if self.camera_mode not in {"concat", "stack"}:
            raise ValueError(f"camera_mode must be 'concat' or 'stack', got {self.camera_mode!r}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        self.rows: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_line_no"] = line_no
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def batch_group_key(self, idx: int) -> tuple:
        row = self.rows[idx]
        cameras = tuple(row.get("camera_names") or ["main"])
        action_type = str(row.get("action_type") or ("none" if row.get("action_start") is None else "dataset_native"))
        action_dim = int(row.get("action_dim", self.action_dim))
        return (self.camera_mode, cameras, self.image_size, action_type, action_dim)

    def source_key(self, idx: int) -> str:
        row = self.rows[idx]
        return str(row.get("source_dataset") or row.get("dataset_family") or "unknown")

    def _frame_path(self, traj_dir: Path, camera: str, frame_idx: int) -> Path:
        frame_name = f"{int(frame_idx):06d}.jpg"
        path = traj_dir / "frames" / camera / frame_name
        if path.exists():
            return path
        legacy = traj_dir / f"rgb_{int(frame_idx)}.png"
        if camera == "main" and legacy.exists():
            return legacy
        raise FileNotFoundError(f"Missing frame for camera={camera} frame={frame_idx}: {path}")

    def _load_video(self, row: dict[str, Any], traj_dir: Path) -> torch.Tensor:
        frame_indices = list(row.get("frame_indices") or range(self.num_video_frames))
        if len(frame_indices) != self.num_video_frames:
            raise ValueError(
                f"Expected {self.num_video_frames} frame_indices, got {len(frame_indices)} in {traj_dir}"
            )
        camera_names = list(row.get("camera_names") or ["main"])
        cams = []
        for camera in camera_names:
            frames = [_load_rgb(self._frame_path(traj_dir, camera, idx), self.image_size) for idx in frame_indices]
            cams.append(torch.stack(frames, dim=1))  # [3,T,H,W]
        if self.camera_mode == "stack":
            return torch.stack(cams, dim=0)  # [V,3,T,H,W]
        return torch.cat(cams, dim=-1) if len(cams) > 1 else cams[0]

    def _load_text_context(self, language: str) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if self.text_embedding_cache_dir is None:
            return None, None
        prompt = language
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        candidates = [
            self.text_embedding_cache_dir / f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt",
            self.text_embedding_cache_dir / f"{hashed}.pt",
        ]
        cache_path = next((p for p in candidates if p.exists()), None)
        if cache_path is None:
            raise FileNotFoundError(
                f"Missing text embedding cache for language={language!r} under {self.text_embedding_cache_dir}"
            )
        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict):
            context = payload.get("context")
            if context is None:
                context = payload.get("embeddings")
            if context is None:
                context = payload.get("hidden_states")
            context_mask = payload.get("context_mask")
            if context_mask is None:
                context_mask = payload.get("mask")
        else:
            context, context_mask = payload, None
        if context is None:
            raise ValueError(f"Invalid text cache payload: {cache_path}")
        context = context.float()
        if context.ndim == 3 and context.shape[0] == 1:
            context = context[0]
        if context_mask is None:
            context_mask = torch.ones(context.shape[0], dtype=torch.bool)
        else:
            context_mask = context_mask.bool()
            if context_mask.ndim == 2 and context_mask.shape[0] == 1:
                context_mask = context_mask[0]
        return context, context_mask

    def _load_trajectory_npz(self, traj_dir: Path) -> dict[str, np.ndarray]:
        path = traj_dir / "trajectory.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}
        # Compatibility with current real rvideo adapters; canonicalization can
        # later consolidate these files into trajectory.npz without changing the loader.
        out = {}
        for key, filename in (("actions", "action.npy"), ("states", "state.npy"), ("video_timestamps", "timestamps.npy")):
            legacy = traj_dir / filename
            if legacy.exists():
                out[key] = np.load(legacy, allow_pickle=False)
        return out

    def _slice_or_pad(self, array: np.ndarray | None, length: int, dim: int, start: int | None = None, end: int | None = None) -> torch.Tensor:
        if array is None or dim <= 0:
            return torch.zeros(length, max(dim, 0), dtype=torch.float32)
        x = np.asarray(array, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if start is not None and end is not None:
            x = x[int(start):int(end)]
        if x.shape[1] != dim:
            if x.shape[1] > dim:
                x = x[:, :dim]
            else:
                pad = np.zeros((x.shape[0], dim - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
        out = torch.zeros(length, dim, dtype=torch.float32)
        take = min(length, x.shape[0])
        if take:
            out[:take] = torch.from_numpy(x[:take])
        return out

    @staticmethod
    def _pair_indices(data, num_items: int) -> dict[tuple[int, int], int]:
        starts = np.asarray(data.get("start_frame_idx", np.arange(num_items)), dtype=np.int64)
        ends = np.asarray(data.get("end_frame_idx", starts + 1), dtype=np.int64)
        return {(int(s), int(e)): i for i, (s, e) in enumerate(zip(starts, ends))}

    @staticmethod
    def _requested_flow_pairs(row: dict[str, Any], num_flow_windows: int) -> list[tuple[int, int]]:
        starts = row.get("flow_start_indices")
        ends = row.get("flow_end_indices")
        if starts is not None and ends is not None:
            return [(int(s), int(e)) for s, e in zip(starts, ends)][:num_flow_windows]
        frames = list(row.get("frame_indices") or [])
        if len(frames) >= 2:
            return [(int(frames[i]), int(frames[i + 1])) for i in range(min(num_flow_windows, len(frames) - 1))]
        return [(i, i + 1) for i in range(num_flow_windows)]

    def _load_flow(self, row: dict[str, Any], traj_dir: Path, loss_mask: dict[str, float]):
        path = traj_dir / "flow" / "teacher_flow.npz"
        if not path.exists():
            loss_mask["vflow"] = 0.0
            loss_mask["aflow"] = 0.0
            loss_mask["mag"] = 0.0
            return (
                torch.zeros(self.num_flow_windows, 3, 3, dtype=torch.float32),
                torch.zeros(self.num_flow_windows, 3, dtype=torch.float32),
                torch.zeros(self.num_flow_windows, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32),
            )
        with np.load(path, allow_pickle=False) as data:
            flow_np = np.asarray(data["flow_vectors"], dtype=np.float32)
            valid_np = np.asarray(data["valid_mask"], dtype=np.float32)
            quality_np = np.asarray(data.get("quality", np.ones(flow_np.shape[0])), dtype=np.float32)
            pair_to_idx = self._pair_indices(data, flow_np.shape[0])
        out_flow = torch.zeros(self.num_flow_windows, 3, 3, dtype=torch.float32)
        out_valid = torch.zeros(self.num_flow_windows, 3, dtype=torch.float32)
        out_quality = torch.zeros(self.num_flow_windows, dtype=torch.float32)
        matched = 0
        for out_i, pair in enumerate(self._requested_flow_pairs(row, self.num_flow_windows)):
            src_i = pair_to_idx.get(pair)
            if src_i is None:
                continue
            out_flow[out_i] = torch.from_numpy(flow_np[src_i])
            out_valid[out_i] = torch.from_numpy(valid_np[src_i])
            out_quality[out_i] = float(quality_np[src_i])
            matched += 1
        if matched == 0:
            loss_mask["vflow"] = 0.0
            loss_mask["aflow"] = 0.0
            loss_mask["mag"] = 0.0
            return out_flow, out_valid, out_quality, torch.tensor(0.0, dtype=torch.float32)
        return out_flow, out_valid, out_quality, torch.tensor(1.0, dtype=torch.float32)

    def _load_grid_flow(self, row: dict[str, Any], traj_dir: Path, loss_mask: dict[str, float]):
        gh, gw = self.grid_size
        if not self.enable_gridflow_teacher:
            loss_mask["gridflow"] = 0.0
            return (
                torch.zeros(self.num_flow_windows, gh, gw, 3, dtype=torch.float32),
                torch.zeros(self.num_flow_windows, gh, gw, dtype=torch.float32),
                torch.zeros(self.num_flow_windows, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32),
            )
        path = traj_dir / "flow" / "grid_flow.npz"
        if not path.exists():
            loss_mask["gridflow"] = 0.0
            return (
                torch.zeros(self.num_flow_windows, gh, gw, 3, dtype=torch.float32),
                torch.zeros(self.num_flow_windows, gh, gw, dtype=torch.float32),
                torch.zeros(self.num_flow_windows, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32),
            )
        if float(loss_mask.get("gridflow", 0.0)) > 0.0:
            meta_path = traj_dir / "flow" / "grid_flow_meta.json"
            grid_meta = _read_json(meta_path) if meta_path.exists() else {}
            if grid_meta.get("grid_assignment") != "reference_frame_projection":
                raise ValueError(
                    f"legacy or invalid grid teacher cannot supervise training: {path}; "
                    "regenerate it with grid_assignment=reference_frame_projection"
                )
        with np.load(path, allow_pickle=False) as data:
            flow_np = np.asarray(data["grid_flow_vectors"], dtype=np.float32)
            valid_np = np.asarray(data["grid_valid_mask"], dtype=np.float32)
            quality_np = np.asarray(data.get("grid_quality", np.ones(flow_np.shape[0])), dtype=np.float32)
            pair_to_idx = self._pair_indices(data, flow_np.shape[0])
        flow = torch.from_numpy(flow_np)
        valid = torch.from_numpy(valid_np)
        if flow.ndim == 3:
            if flow.shape[1] != gh * gw:
                raise ValueError(f"grid_flow_vectors has flattened G={flow.shape[1]}, expected {gh * gw}: {path}")
            flow = flow.view(flow.shape[0], gh, gw, 3)
        if valid.ndim == 2:
            if valid.shape[1] != gh * gw:
                raise ValueError(f"grid_valid_mask has flattened G={valid.shape[1]}, expected {gh * gw}: {path}")
            valid = valid.view(valid.shape[0], gh, gw)
        out_flow = torch.zeros(self.num_flow_windows, gh, gw, 3, dtype=torch.float32)
        out_valid = torch.zeros(self.num_flow_windows, gh, gw, dtype=torch.float32)
        out_quality = torch.zeros(self.num_flow_windows, dtype=torch.float32)
        matched = 0
        for out_i, pair in enumerate(self._requested_flow_pairs(row, self.num_flow_windows)):
            src_i = pair_to_idx.get(pair)
            if src_i is None:
                continue
            out_flow[out_i] = flow[src_i, :gh, :gw]
            out_valid[out_i] = valid[src_i, :gh, :gw]
            out_quality[out_i] = float(quality_np[src_i])
            matched += 1
        if matched == 0:
            loss_mask["gridflow"] = 0.0
            return out_flow, out_valid, out_quality, torch.tensor(0.0, dtype=torch.float32)
        return out_flow, out_valid, out_quality, torch.tensor(1.0, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        traj_dir = Path(row["traj_dir"])
        meta = _read_json(traj_dir / "meta.json") if (traj_dir / "meta.json").exists() else {}
        semantics_path = traj_dir / "semantics.json"
        if not semantics_path.exists():
            semantics_path = traj_dir / "llm_semantics.json"
        semantics = _read_json(semantics_path) if semantics_path.exists() else {}
        language_path = traj_dir / "language.txt"
        language = language_path.read_text(encoding="utf-8").strip() if language_path.exists() else semantics.get("language", "")
        video = self._load_video(row, traj_dir)

        loss_mask = {k: float(row.get("loss_mask", {}).get(k, 0.0)) for k in LOSS_KEYS}
        has_action = bool(meta.get("has_action", row.get("has_action", False))) and row.get("action_start") is not None
        traj = self._load_trajectory_npz(traj_dir)
        if has_action:
            action = self._slice_or_pad(
                traj.get("actions"), self.num_action_steps, self.action_dim, row.get("action_start"), row.get("action_end")
            )
            action_mask = torch.tensor(1.0, dtype=torch.float32)
            action_is_pad = torch.zeros(self.num_action_steps, dtype=torch.bool)
        else:
            action = torch.zeros(self.num_action_steps, self.action_dim, dtype=torch.float32)
            action_mask = torch.tensor(0.0, dtype=torch.float32)
            action_is_pad = torch.ones(self.num_action_steps, dtype=torch.bool)
            loss_mask["action"] = 0.0
            loss_mask["aflow"] = 0.0

        state = self._slice_or_pad(traj.get("states"), self.num_video_frames, self.state_dim)
        proprio = self._slice_or_pad(traj.get("proprio"), self.num_action_steps, self.proprio_dim)
        state_mask = torch.tensor(float(self.state_dim > 0 and "states" in traj), dtype=torch.float32)
        proprio_mask = torch.tensor(float(self.proprio_dim > 0 and "proprio" in traj), dtype=torch.float32)

        iflow_teacher, iflow_valid, iflow_quality, iflow_mask = self._load_flow(row, traj_dir, loss_mask)
        grid_flow_teacher, grid_flow_valid, grid_flow_quality, grid_flow_mask = self._load_grid_flow(row, traj_dir, loss_mask)
        prompt = self.prompt_template.format(task=language)
        context, context_mask = self._load_text_context(prompt)

        sample = {
            "video": video,
            "language": language,
            "prompt": prompt,
            "semantics": semantics,
            "action": action,
            "action_mask": action_mask,
            "action_is_pad": action_is_pad,
            "image_is_pad": torch.zeros(self.num_video_frames, dtype=torch.bool),
            "state": state,
            "state_mask": state_mask,
            "proprio": proprio,
            "proprio_mask": proprio_mask,
            "proprio_is_pad": torch.zeros(self.num_action_steps, dtype=torch.bool),
            "iflow_teacher": iflow_teacher,
            "iflow_valid": iflow_valid,
            "iflow_quality": iflow_quality,
            "iflow_mask": iflow_mask,
            "grid_flow_teacher": grid_flow_teacher,
            "grid_flow_valid": grid_flow_valid,
            "grid_flow_quality": grid_flow_quality,
            "grid_flow_mask": grid_flow_mask,
            "loss_mask": loss_mask,
            "dataset_family": row.get("dataset_family"),
            "source_dataset": row.get("source_dataset"),
            "task_label": row.get("task_label"),
            "traj_dir": str(traj_dir),
            "layout_key": repr(self.batch_group_key(idx)),
        }
        if context is not None:
            sample["context"] = context
            sample["context_mask"] = context_mask
        return sample

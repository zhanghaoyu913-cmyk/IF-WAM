import logging
import json
import inspect
import os
import re
import subprocess
import sys
from collections import Counter
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from ifwam.data import LayoutHomogeneousBatchSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_weights_every = self._resolve_interval_cfg("save_weights_every", fallback=self.save_every)
        self.save_weights_steps = self._resolve_steps_cfg("save_weights_steps")
        self.save_state_every = self._resolve_interval_cfg("save_state_every", fallback=self.save_every)
        self.save_final_state = bool(cfg.get("save_final_state", True))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.trainable_scope = str(cfg.get("trainable_scope", "dit"))
        self.expected_global_batch_size = cfg.get("expected_global_batch_size", None)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self.wandb_core_metrics_only = bool(cfg.wandb.get("core_metrics_only", False))
        self._last_active_loss_metrics = {}

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        if deepspeed_plugin is not None:
            # A DataLoader backed by a custom batch_sampler reports
            # `batch_size=None`, so Accelerate cannot resolve DeepSpeed's
            # automatic batch-size fields from the loader itself.
            deepspeed_config = deepspeed_plugin.deepspeed_config
            deepspeed_config["train_micro_batch_size_per_gpu"] = self.batch_size
            deepspeed_config["gradient_accumulation_steps"] = self.gradient_accumulation_steps
            deepspeed_config["train_batch_size"] = (
                self.batch_size
                * self.gradient_accumulation_steps
                * self.accelerator.num_processes
            )
        zero_stage = (
            deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "disabled")
            if deepspeed_plugin is not None
            else "disabled"
        )
        self.global_batch_size = self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes
        if self.expected_global_batch_size is not None and int(self.expected_global_batch_size) != int(self.global_batch_size):
            raise ValueError(
                f"global_batch_size mismatch: expected {self.expected_global_batch_size}, got {self.global_batch_size} "
                f"(per_device={self.batch_size}, grad_accum={self.gradient_accumulation_steps}, world_size={self.accelerator.num_processes})"
            )

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d global_batch=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.global_batch_size,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)

        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # DiT and optional IF-WAM side heads remain trainable.
        self._apply_dit_only_train_mode(self.model, trainable_scope=self.trainable_scope)
        trainable_params = self._collect_trainable_params(self.model)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)
        self.train_metrics_path = os.path.join(self.output_dir, "train_metrics.jsonl")
        self.actual_seen_by_source = Counter()
        self.actual_seen_by_layout = Counter()
        if self.accelerator.is_main_process:
            self._write_run_metadata()
            self._write_data_and_sampler_stats()

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _resolve_interval_cfg(self, name: str, *, fallback: int) -> int:
        value = self.cfg.get(name, None)
        if value is None or str(value).strip().lower() in {"", "none", "null"}:
            return int(fallback)
        return int(value)

    def _resolve_steps_cfg(self, name: str) -> set[int]:
        value = self.cfg.get(name, None)
        if value is None or str(value).strip().lower() in {"", "none", "null"}:
            return set()
        if isinstance(value, str):
            items = [part.strip() for part in value.split(",") if part.strip()]
        else:
            items = list(value)
        return {int(item) for item in items}

    def _should_run_interval(self, interval: int) -> bool:
        return interval > 0 and self.global_step > 0 and self.global_step % interval == 0

    def _display_loss_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Carry forward inactive branch losses for readable logs only.

        The raw training loss remains unchanged.  Mixed IF-WAM batches can have
        vflow/aflow/gridflow masks equal to zero; logging those branch losses as
        zero creates artificial drops in W&B, so display metrics reuse the last
        active value while mask ratios still expose which branch contributed.
        """
        display = dict(metrics)
        branches = {
            "vflow": ["loss_vflow", "ifwam/vflow_dir", "ifwam/vflow_mag", "ifwam/vflow_rel", "ifwam/vflow_static", "ifwam/vflow_consistency"],
            "aflow": ["loss_aflow", "ifwam/aflow_dir_rel", "ifwam/aflow_agent_dir", "ifwam/aflow_rel", "ifwam/aflow_contrastive", "ifwam/aflow_consistency", "ifwam/aflow_sigma_weight"],
            "gridflow": ["loss_gridflow", "ifwam/grid_motion_presence", "ifwam/grid_presence_pos", "ifwam/grid_presence_neg", "ifwam/grid_moving_cell_ratio", "ifwam/grid_move_threshold", "ifwam/grid_presence_recall", "ifwam/grid_presence_precision", "ifwam/grid_dir", "ifwam/grid_smooth", "ifwam/grid_mag"],
        }
        action_ratio = float(metrics.get("ifwam/loss_mask_action_ratio", 1.0))
        if "loss_action" in metrics:
            display["raw/loss_action_active"] = metrics["loss_action"]
            if action_ratio > 0.0:
                self._last_active_loss_metrics["loss_action_active"] = metrics["loss_action"]
                display["loss_action_active"] = metrics["loss_action"]
            elif "loss_action_active" in self._last_active_loss_metrics:
                display["loss_action_active"] = self._last_active_loss_metrics["loss_action_active"]

        for branch, keys in branches.items():
            ratio = float(metrics.get(f"ifwam/loss_mask_{branch}_ratio", 1.0))
            active = ratio > 0.0
            for key in keys:
                if key not in metrics:
                    continue
                raw_key = f"raw/{key}"
                display[raw_key] = metrics[key]
                if active:
                    self._last_active_loss_metrics[key] = metrics[key]
                elif key in self._last_active_loss_metrics:
                    display[key] = self._last_active_loss_metrics[key]
        return display

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None


    def _git_commit_hash(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                check=True,
                text=True,
                capture_output=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def _dataset_source_counts(self) -> dict[str, int]:
        counts = Counter()
        for idx in range(len(self.train_dataset)):
            if hasattr(self.train_dataset, "source_key"):
                key = self.train_dataset.source_key(idx)
            else:
                key = "unknown"
            counts[str(key)] += 1
        return dict(sorted(counts.items()))

    def _dataset_layout_counts(self) -> dict[str, int]:
        counts = Counter()
        if hasattr(self.train_dataset, "batch_group_key"):
            for idx in range(len(self.train_dataset)):
                counts[repr(self.train_dataset.batch_group_key(idx))] += 1
        return dict(sorted(counts.items()))

    def _write_run_metadata(self):
        payload = {
            "git_commit_hash": self._git_commit_hash(),
            "command_line": sys.argv,
            "output_dir": self.output_dir,
            "global_batch_size": int(self.global_batch_size),
            "per_device_batch_size": int(self.batch_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "world_size": int(self.accelerator.num_processes),
            "seed": int(self.seed),
        }
        with open(os.path.join(self.output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _write_data_and_sampler_stats(self):
        data_stats = {
            "dataset_num_rows": int(len(self.train_dataset)),
            "source_counts": self._dataset_source_counts(),
            "layout_counts": self._dataset_layout_counts(),
        }
        sampler_stats = {
            "sampler_class": type(self.train_sampler).__name__,
            "batch_size_per_device": int(self.batch_size),
            "global_batch_size": int(self.global_batch_size),
            "drop_last": bool(getattr(self.train_sampler, "drop_last", False)),
            "shuffle": bool(getattr(self.train_sampler, "shuffle", False)),
            "group_sampling": str(getattr(self.train_sampler, "group_sampling", "unknown")),
            "source_weights": self.train_sampler.effective_source_weights() if hasattr(self.train_sampler, "effective_source_weights") else {},
            "num_batches_per_epoch": int(len(self.train_sampler)) if hasattr(self.train_sampler, "__len__") else None,
            "group_counts": self.train_sampler.group_counts() if hasattr(self.train_sampler, "group_counts") else {},
        }
        with open(os.path.join(self.output_dir, "data_stats.json"), "w", encoding="utf-8") as f:
            json.dump(data_stats, f, ensure_ascii=True, indent=2)
        with open(os.path.join(self.output_dir, "sampler_stats.json"), "w", encoding="utf-8") as f:
            json.dump(sampler_stats, f, ensure_ascii=True, indent=2)
        logger.info("Data source counts: %s", data_stats["source_counts"])
        logger.info("Sampler group counts: %s", sampler_stats["group_counts"])

    def _sync_counter(self, counter: Counter) -> Counter:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered, dict(counter))
            merged = Counter()
            for item in gathered:
                merged.update(item or {})
            return merged
        return Counter(counter)

    def _batch_counter(self, sample, key: str) -> Counter:
        values = sample.get(key, [])
        if isinstance(values, str):
            values = [values]
        return Counter(str(v) for v in values)

    def _append_train_metrics(self, payload: dict):
        with open(self.train_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")

    def _build_loader(self, dataset, worker_init_fn=None):
        collate_fn = getattr(dataset, "collate_fn", None)
        if bool(getattr(dataset, "layout_homogeneous_batches", False)):
            self.train_sampler = LayoutHomogeneousBatchSampler(
                dataset=dataset,
                batch_size=self.batch_size,
                seed=self.seed,
                drop_last=bool(getattr(dataset, "sampler_drop_last", True)),
                group_sampling=str(getattr(dataset, "sampler_group_sampling", "proportional_to_num_rows")),
                shuffle=bool(getattr(dataset, "sampler_shuffle", True)),
            )
            return DataLoader(
                dataset,
                batch_sampler=self.train_sampler,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=worker_init_fn,
                collate_fn=collate_fn,
            )
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting trainable_scope=%s and freezing other model components.", self.trainable_scope)
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model, trainable_scope=self.trainable_scope)

    @staticmethod
    def _apply_dit_only_train_mode(model, trainable_scope: str = "dit"):
        model.eval()
        model.requires_grad_(False)
        def _enable(module):
            if module is not None:
                module.train()
                module.requires_grad_(True)

        if trainable_scope == "dit":
            model.dit.train()
            model.dit.requires_grad_(True)
            for name in ("proprio_encoder", "process_flow_readout", "flow_scoring_head", "grid_expert", "grid_aux_decoder"):
                _enable(getattr(model, name, None))
            if getattr(model, "grid_aux_arch", None) == "cross_attn_decoder":
                grid_expert = getattr(model, "grid_expert", None)
                grid_blocks = getattr(grid_expert, "blocks", None) if grid_expert is not None else None
                if grid_blocks is not None:
                    grid_blocks.eval()
                    grid_blocks.requires_grad_(False)
        elif trainable_scope == "action_head_only":
            action_expert = getattr(model, "action_expert", None)
            action_head = getattr(action_expert, "head", None) if action_expert is not None else None
            if action_head is None:
                raise RuntimeError("trainable_scope=action_head_only requires model.action_expert.head.")
            action_head.train()
            action_head.requires_grad_(True)
        elif trainable_scope == "action_adapter_only":
            action_expert = getattr(model, "action_expert", None)
            if action_expert is None:
                raise RuntimeError("trainable_scope=action_adapter_only requires model.action_expert.")
            selected = []
            for name in ("action_encoder", "time_embedding", "time_projection", "head"):
                module = getattr(action_expert, name, None)
                if module is not None:
                    selected.append(module)
            proprio_encoder = getattr(model, "proprio_encoder", None)
            if proprio_encoder is not None:
                selected.append(proprio_encoder)
            if not selected:
                raise RuntimeError("trainable_scope=action_adapter_only found no action adapter modules.")
            for module in selected:
                _enable(module)
        else:
            raise ValueError(f"Unsupported trainable_scope={trainable_scope!r}")

    @staticmethod
    def _collect_trainable_params(model):
        modules = [model.dit]
        for name in ("action_expert", "proprio_encoder", "process_flow_readout", "flow_scoring_head", "grid_expert", "grid_aux_decoder"):
            module = getattr(model, name, None)
            if module is not None:
                modules.append(module)
        params = []
        seen = set()
        for module in modules:
            for param in module.parameters():
                if param.requires_grad and id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
        if not params:
            raise RuntimeError("No trainable parameters were selected.")
        logger.info(
            "Selected %d trainable tensors (%.3f B parameters) from DiT and optional IF-WAM heads.",
            len(params),
            sum(param.numel() for param in params) / 1e9,
        )
        return params

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self, *, save_weights: bool = True, save_state: bool = True):
        step_tag = f"step_{self.global_step:06d}"

        ckpt_path = None
        if save_weights:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
            self.accelerator.wait_for_everyone()

        state_path = None
        if save_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        accum_loss_sum = torch.tensor(0.0, device=self.accelerator.device, dtype=torch.float32)
        accum_metric_sums: dict[str, float] = {}
        accum_source_counts = Counter()
        accum_layout_counts = Counter()
        accum_micro_count = 0

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                accum_loss_sum = accum_loss_sum + loss.detach().float()
                for key, value in loss_dict.items():
                    accum_metric_sums[key] = accum_metric_sums.get(key, 0.0) + float(value)
                accum_source_counts.update(self._batch_counter(sample, "source_dataset"))
                accum_layout_counts.update(self._batch_counter(sample, "layout_key"))
                accum_micro_count += 1
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    denom_micro = max(1, accum_micro_count)
                    loss_for_log = accum_loss_sum / float(denom_micro)
                    metrics_for_log = {key: value / float(denom_micro) for key, value in accum_metric_sums.items()}
                    accum_loss_sum = torch.tensor(0.0, device=self.accelerator.device, dtype=torch.float32)
                    source_counts_for_step = self._sync_counter(accum_source_counts)
                    layout_counts_for_step = self._sync_counter(accum_layout_counts)
                    self.actual_seen_by_source.update(source_counts_for_step)
                    self.actual_seen_by_layout.update(layout_counts_for_step)
                    accum_metric_sums = {}
                    accum_source_counts = Counter()
                    accum_layout_counts = Counter()
                    accum_micro_count = 0
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss_for_log.reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in metrics_for_log.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        display_loss_metrics = self._display_loss_metrics(global_loss_metrics)
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if display_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(display_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        total_samples_seen = int(self.global_step * self.global_batch_size)
                        effective_epoch = float(total_samples_seen / max(len(self.train_dataset), 1))
                        unweighted_grid_loss = float(global_loss_metrics.get(
                            "unweighted_gridflow_loss",
                            global_loss_metrics.get("unweighted_gridflow_fm_loss", global_loss_metrics.get("loss_grid_raw", 0.0)),
                        ))
                        weighted_grid_loss = float(global_loss_metrics.get(
                            "weighted_gridflow_loss",
                            global_loss_metrics.get("weighted_gridflow_fm_loss", global_loss_metrics.get("loss_grid_scaled", 0.0)),
                        ))
                        train_metrics_payload = {
                            "global_step": int(self.global_step),
                            "global_batch_size": int(self.global_batch_size),
                            "dataset_num_rows": int(len(self.train_dataset)),
                            "effective_epoch": effective_epoch,
                            "total_samples_seen": total_samples_seen,
                            "actual_seen_by_source": dict(sorted(self.actual_seen_by_source.items())),
                            "actual_seen_by_layout": dict(sorted(self.actual_seen_by_layout.items())),
                            "action_mask_fraction": float(global_loss_metrics.get("ifwam/loss_mask_action_ratio", 0.0)),
                            "gridflow_mask_fraction": float(global_loss_metrics.get("ifwam/loss_mask_gridflow_ratio", 0.0)),
                            "unweighted_action_loss": float(global_loss_metrics.get("unweighted_action_loss", 0.0)),
                            "unweighted_gridflow_loss": unweighted_grid_loss,
                            "weighted_action_loss": float(global_loss_metrics.get("weighted_action_loss", 0.0)),
                            "weighted_gridflow_loss": weighted_grid_loss,
                            "total_loss": float(global_loss_metrics.get("total_loss", global_loss)),
                            "grad_norm": global_grad_norm,
                            "lr": current_lr,
                        }
                        for key in (
                            "action_prediction_norm",
                            "action_target_norm",
                            "loss_grid_raw",
                            "loss_grid_scaled",
                            "loss_grid_direction",
                            "loss_grid_presence",
                            "grid_teacher_sample_coverage",
                            "grid_valid_cell_ratio",
                            "grid_moving_cell_ratio",
                            "grid_quality_mean",
                            "grid_flow_norm_mean_moving",
                            "grid_flow_norm_p50_moving",
                            "grid_lambda_effective",
                            "num_grid_tokens_valid",
                        ):
                            if key in global_loss_metrics:
                                train_metrics_payload[key] = float(global_loss_metrics[key])
                        self._append_train_metrics(train_metrics_payload)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in display_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        if self.wandb_core_metrics_only:
                            core_keys = {
                                "train/loss",
                                "train/loss_video",
                                "train/loss_action",
                                "train/loss_action_active",
                                "train/loss_gridflow",
                                "train/ifwam/grid_moving_cell_ratio",
                                "train/ifwam/grid_move_threshold",
                                "train/ifwam/grid_dir",
                                "train/ifwam/grid_smooth",
                                "train/ifwam/loss_mask_action_ratio",
                                "train/ifwam/loss_mask_gridflow_ratio",
                            }
                            wandb_payload = {k: v for k, v in wandb_payload.items() if k in core_keys}
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    save_weights = self._should_run_interval(self.save_weights_every) or self.global_step in self.save_weights_steps
                    save_state = self._should_run_interval(self.save_state_every)
                    if save_weights or save_state:
                        ckpt_info = self.save_checkpoint(save_weights=save_weights, save_state=save_state)
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint(save_weights=True, save_state=self.save_final_state)
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint(save_weights=True, save_state=self.save_final_state)
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        

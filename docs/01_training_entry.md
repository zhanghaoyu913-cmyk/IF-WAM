# FastWAM 训练入口与代码路径

## 1. 训练启动入口

当前仓库的训练入口是 `scripts/train.py`，它只做两件事：

1. 注册 Hydra resolver。
2. 调用 `fastwam.runtime.run_training(cfg)`。

实际多卡启动脚本是：

```bash
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

调用链：

```text
scripts/train_zero1.sh
  -> accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml scripts/train.py ...
    -> scripts/train.py:main(cfg)
      -> fastwam.runtime.run_training(cfg)
        -> instantiate(cfg.model, ...)
        -> build_datasets(cfg.data)
        -> Wan22Trainer(...).train()
```

`scripts/train_zero1.sh` 还会根据 `task=...` 自动设置输出目录：

```text
runs/{task_name}/{run_id}/
  config.yaml
  dataset_stats.json
  checkpoints/
    weights/step_*.pt
    state/step_*/
  eval/
```

## 2. Hydra 配置链路

根配置是 `configs/train.yaml`：

```yaml
defaults:
  - _self_
  - data: null
  - model: null
  - task: null
```

具体任务通过 `configs/task/*.yaml` 覆盖 `data` 和 `model`。例如：

| task | data | model | 说明 |
|---|---|---|---|
| `libero_uncond_2cam224_1e-4` | `libero_2cam` | `fastwam` | LIBERO 2 相机，基础 FastWAM |
| `libero_joint_2cam224_1e-4` | `libero_2cam` | `fastwam_joint` | action 可看完整 video latent |
| `libero_idm_2cam224_1e-4` | `libero_2cam` | `fastwam_idm` | teacher-forcing IDM 变体 |
| `robotwin_uncond_3cam_384_1e-4` | `robotwin` | `fastwam` | RoboTwin 3 相机拼接，基础 FastWAM |

模型配置入口：

```text
configs/model/fastwam.yaml       -> fastwam.runtime.create_fastwam
configs/model/fastwam_joint.yaml -> fastwam.runtime.create_fastwam_joint
configs/model/fastwam_idm.yaml   -> fastwam.runtime.create_fastwam_idm
```

三个模型配置共享主要结构：

- Wan2.2 TI2V 5B 的视频 DiT 作为 `video_expert`。
- 自定义 `ActionDiT` 作为 `action_expert`。
- `MoT` 把 video/action 两个 expert 的 self-attention 混合起来。
- Wan2.2 VAE 编码/解码视频 latent。
- T5 文本编码训练时通常预计算为 cache，推理/评测时可加载文本编码器。
- `proprio_encoder` 把机器人状态投影为一个额外 context token。

## 3. 训练前置步骤

### 3.1 预处理 ActionDiT backbone

入口：`scripts/preprocess_action_dit_backbone.py`

用途：从 WanVideoDiT 的 backbone 权重插值/缩放出 ActionDiT backbone 权重，保存成：

```text
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

ActionDiT 的 `action_encoder` 和 `head` 不从视频 DiT 复制，属于随机初始化或后续 checkpoint 覆盖。

### 3.2 预计算文本 embedding

入口：`scripts/precompute_text_embeds.py`

用途：读取 `data.*.dataset_dirs/meta/tasks.jsonl` 中的任务文本，套用模板：

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

然后用 Wan2.2 T5 encoder 生成 cache：

```text
data/text_embeds_cache/{dataset}/{sha256}.t5_len128.wan22ti2v5b.pt
```

训练配置里 `model.load_text_encoder=false`，所以训练样本必须提供 `context/context_mask`，即依赖这个 cache。

## 4. 数据集与 batch 构造

数据入口：`src/fastwam/datasets/lerobot/robot_video_dataset.py`

处理器入口：`src/fastwam/datasets/lerobot/processors/fastwam_processor.py`

数据流：

```text
BaseLerobotDataset raw sample
  -> FastWAMProcessor.preprocess
     - instruction 选择/增强
     - 每个相机图像 ToTensor + Resize
     - action/state transform
     - action/state normalizer
     - ConcatLeftAlign 合并 action/state 维度
  -> RobotVideoDataset._get
     - 按 action_video_freq_ratio 抽视频帧
     - 多相机拼接
     - resize/crop/Normalize 到 [-1,1]
     - 读取 cached text context
  -> DataLoader batch
```

当前典型配置：

| 数据集 | 原始步数 `num_frames` | `action_video_freq_ratio` | 视频帧数 | action horizon |
|---|---:|---:|---:|---:|
| LIBERO | 33 | 4 | 9 | 32 |
| RoboTwin | 33 | 4 | 9 | 32 |

注意：`action_horizon=32`，而视频只有 9 帧，因此每两个视频帧之间对应 4 个 action step。

## 5. `run_training` 做了什么

入口：`src/fastwam/runtime.py::run_training`

核心步骤：

1. 设置日志并注册 `cfg.output_dir` 为工作目录。
2. 保存解析后的 `config.yaml`。
3. 根据 `mixed_precision` 选择模型 dtype。
4. `instantiate(cfg.model, model_dtype=..., device=...)` 创建模型。
5. `build_datasets(cfg.data)` 创建 train/val dataset。
6. 构造 `Wan22Trainer` 并调用 `trainer.train()`。

## 6. Trainer 训练循环

入口：`src/fastwam/trainer.py::Wan22Trainer`

训练初始化重点：

- 使用 `Accelerator` 支持多卡、DeepSpeed、混合精度和梯度累积。
- 冻结非训练模块：`model.eval()` 和 `model.requires_grad_(False)`。
- 只把 `model.dit` 设为 train 且可训练。FastWAM 中 `model.dit` 是 `self.mot`，因此训练的是 MoT 内的 video/action experts。
- 如果启用了 `proprio_encoder`，它也参与训练。
- Optimizer 是 AdamW，参数来自 `model.dit.parameters()` 和可选 `proprio_encoder.parameters()`。

单步训练调用链：

```text
Wan22Trainer.train
  -> sample = next(DataLoader)
  -> train_model.training_loss(sample)
     -> FastWAM.build_inputs(sample)
     -> VAE encode video
     -> sample video/action diffusion timestep
     -> add noise
     -> video_expert.pre_dit(...)
     -> action_expert.pre_dit(...)
     -> MoT(...)
     -> video_expert.post_dit(...)
     -> action_expert.post_dit(...)
     -> loss_video + loss_action
  -> accelerator.backward(loss)
  -> clip grad
  -> optimizer.step()
  -> scheduler.step()
  -> log/eval/checkpoint
```

## 7. 损失计算

当前 FastWAM 系列都用连续 flow matching scheduler：

```text
noisy = (1 - sigma) * sample + sigma * noise
target = noise - sample
```

视频损失：

- 在 VAE latent 空间预测 `target_video`。
- 如果第一帧 latent 被固定为输入图像 latent，则 loss 去掉第一个 latent time step。
- 如果有 `image_is_pad`，会按 VAE 时间下采样关系把 pad mask 映射到 latent time。

动作损失：

- 在 normalized action 空间预测 `target_action`。
- 如果有 `action_is_pad`，只对有效 action step 求平均。

总损失：

```text
loss_total = lambda_video * loss_video + lambda_action * loss_action
```

`configs/model/*.yaml` 显式配置了 `lambda_action: 1.0`，`lambda_video` 没写时由代码默认取 `1.0`。

## 8. 推理与评测入口

### 8.1 通用 runtime 推理

`src/fastwam/runtime.py::run_inference` 会：

1. `instantiate(cfg.model, load_text_encoder=...)`
2. 可选加载 checkpoint。
3. 读取单张输入图像。
4. 调用 `model.infer(...)`。
5. 保存 mp4。

这个函数在当前仓库中更像通用工具，主要 benchmark 入口在 `experiments/` 下。

### 8.2 LIBERO 评测

入口链路：

```text
experiments/libero/run_libero_manager.py
  -> experiments/libero/eval_libero_single.py
     -> instantiate(cfg.model, load_text_encoder=true)
     -> model.load_checkpoint(cfg.ckpt)
     -> processor.set_normalizer_from_stats(dataset_stats)
     -> _predict_action_chunk(...)
        -> model.infer_action(...)
        或 visualize_future_video=true 时 model.infer_joint(...)
```

LIBERO 会把当前 obs 的主视角和腕部相机 resize 后横向拼接为 `[1,3,224,448]`，再生成 action chunk。

### 8.3 RoboTwin 评测

入口链路：

```text
experiments/robotwin/run_robotwin_manager.py
  -> third_party/RoboTwin policy fastwam_policy
     -> experiments/robotwin/fastwam_policy/deploy_policy.py
        -> WorldActionRobotWinPolicy._infer_action_chunk
           -> model.infer_action(...)
```

RoboTwin 把三路相机拼成 `[1,3,384,320]`：

```text
top camera:  256 x 320
left wrist: 128 x 160
right wrist:128 x 160
bottom = concat(left, right): 128 x 320
final = concat(top, bottom): 384 x 320
```

## 9. 入口代码速查

| 目标 | 文件 |
|---|---|
| 训练 main | `scripts/train.py` |
| Accelerate/DeepSpeed 启动 | `scripts/train_zero1.sh`, `scripts/train_zero2.sh` |
| 构建模型/数据/训练 | `src/fastwam/runtime.py` |
| Trainer | `src/fastwam/trainer.py` |
| 数据集 | `src/fastwam/datasets/lerobot/robot_video_dataset.py` |
| Processor | `src/fastwam/datasets/lerobot/processors/fastwam_processor.py` |
| FastWAM 基类 | `src/fastwam/models/wan22/fastwam.py` |
| Joint 变体 | `src/fastwam/models/wan22/fastwam_joint.py` |
| IDM 变体 | `src/fastwam/models/wan22/fastwam_idm.py` |
| 视频 DiT expert | `src/fastwam/models/wan22/wan_video_dit.py` |
| Action DiT expert | `src/fastwam/models/wan22/action_dit.py` |
| Mixture-of-Transformers | `src/fastwam/models/wan22/mot.py` |


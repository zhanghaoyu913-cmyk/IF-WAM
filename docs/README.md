# FastWAM 代码分析文档索引

本目录是根据当前仓库代码整理的训练入口、模型实现、推理路径和张量尺度说明。

- [01_training_entry.md](01_training_entry.md): 训练入口、Hydra 配置链路、Trainer 训练循环、评测/推理入口。
- [02_model_architecture.html](02_model_architecture.html): FastWAM 模型块、训练与推理流程、三个模型变体的直观框架图。
- [03_tensor_shapes.md](03_tensor_shapes.md) / [03_tensor_shapes.html](03_tensor_shapes.html): 数据输入后每个主要模块的尺度变化、符号含义、LIBERO/RoboTwin 典型尺寸表。

核心代码路径：

- `scripts/train.py`
- `scripts/train_zero1.sh`
- `src/fastwam/runtime.py`
- `src/fastwam/trainer.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `src/fastwam/datasets/lerobot/processors/fastwam_processor.py`
- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/fastwam_joint.py`
- `src/fastwam/models/wan22/fastwam_idm.py`
- `src/fastwam/models/wan22/wan_video_dit.py`
- `src/fastwam/models/wan22/action_dit.py`
- `src/fastwam/models/wan22/mot.py`

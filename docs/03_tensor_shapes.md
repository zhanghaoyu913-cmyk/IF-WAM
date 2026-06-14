# FastWAM 张量尺度与数据流速查

## 1. 符号表

| 符号 | 含义 | 当前典型值 |
|---|---|---|
| `B` | batch size | 训练配置常见为 16 |
| `N_raw` | 数据集连续观测步数，`data.train.num_frames` | 33 |
| `r` | 视频抽帧间隔，`action_video_freq_ratio` | 4 |
| `T` | 输入模型的视频帧数 | `(N_raw - 1) / r + 1 = 9` |
| `A` | action horizon | `N_raw - 1 = 32` |
| `H,W` | 多相机拼接后的最终视频高宽 | LIBERO: 224x448, RoboTwin: 384x320 |
| `Da` | action 维度 | LIBERO: 7, RoboTwin: 14 |
| `Dp` | proprio/state 维度 | LIBERO: 8, RoboTwin: 14 |
| `L_text` | T5 文本 context 长度 | 128 |
| `L` | text + 可选 proprio token 长度 | `128` 或 `129` |
| `Cv` | Wan2.2 VAE latent channel | 48 |
| `Tz` | VAE latent 时间长度 | `(T - 1) / 4 + 1 = 3` |
| `Hz,Wz` | VAE latent 空间尺寸 | `H/16, W/16` |
| `p` | VideoDiT patch size | `[1,2,2]` |
| `Sv` | video token 数 | `Tz * (Hz/2) * (Wz/2)` |
| `Sa` | action token 数 | `A` |
| `Dv` | video expert hidden dim | 3072 |
| `Dh_action` | action expert hidden dim | 1024 |
| `Hh` | attention heads | 24 |
| `Dhead` | attention head dim | 128 |
| `Dqkv` | mixed attention Q/K/V 维度 | `Hh * Dhead = 3072` |

## 2. 数据集输出尺度

`RobotVideoDataset._get` 返回单样本，`DataLoader` collate 后加上 batch 维。

### 单样本

| 字段 | 单样本尺度 | 含义 |
|---|---|---|
| `video` | `[3,T,H,W]` | 拼接相机后的视频，范围 [-1,1] |
| `action` | `[A,Da]` | 归一化动作序列 |
| `proprio` | `[A,Dp]` | 归一化状态，代码中取 `sample["proprio"][:-1]` 与 action 对齐 |
| `context` | `[128,4096]` | 预计算 T5 text embedding |
| `context_mask` | `[128]` | cache 中读取后代码会置为全 1 |
| `image_is_pad` | `[T]` | 视频帧 pad 标记 |
| `action_is_pad` | `[A]` | action pad 标记 |

### Batch

| 字段 | batch 尺度 |
|---|---|
| `video` | `[B,3,T,H,W]` |
| `action` | `[B,A,Da]` |
| `proprio` | `[B,A,Dp]` |
| `context` | `[B,128,4096]` |
| `context_mask` | `[B,128]` |
| `image_is_pad` | `[B,T]` |
| `action_is_pad` | `[B,A]` |

## 3. 当前任务的具体尺度

### LIBERO 2 相机配置

来自 `configs/data/libero_2cam.yaml`：

- 两路相机：`image` 和 `wrist_image`。
- 每路先 resize 到 `[3,224,224]`。
- `concat_multi_camera="horizontal"`，所以最终视频是 `H=224, W=448`。
- action 维度 `Da=7`，proprio 维度 `Dp=8`。

| 阶段 | 尺度 |
|---|---|
| 原始 processor 图像 | 2 cameras，每个 `[T,3,224,224]` |
| 拼接后视频 | `[B,3,9,224,448]` |
| VAE latent | `[B,48,3,14,28]` |
| VideoDiT patch grid | `F=3, h=7, w=14` |
| video tokens | `[B,294,3072]` |
| 每个 latent frame 的 token 数 | `7 * 14 = 98` |
| action | `[B,32,7]` |
| action tokens | `[B,32,1024]` |
| text context | `[B,128,4096]` |
| text + proprio context | `[B,129,4096]` |

### RoboTwin 3 相机配置

来自 `configs/data/robotwin.yaml`：

- 三路相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist`。
- `concat_multi_camera="robotwin"`：
  - 顶部相机 resize 到 `256x320`。
  - 左/右腕部相机各 resize 到 `128x160`。
  - 左右横向拼成 `128x320`，再和顶部纵向拼成 `384x320`。
- action/proprio 维度都是 14。

| 阶段 | 尺度 |
|---|---|
| 拼接后视频 | `[B,3,9,384,320]` |
| VAE latent | `[B,48,3,24,20]` |
| VideoDiT patch grid | `F=3, h=12, w=10` |
| video tokens | `[B,360,3072]` |
| 每个 latent frame 的 token 数 | `12 * 10 = 120` |
| action | `[B,32,14]` |
| action tokens | `[B,32,1024]` |
| text context | `[B,128,4096]` |
| text + proprio context | `[B,129,4096]` |

## 4. 训练前向传播尺度

以下以当前 FastWAM 系列共同路径为主。

### 4.1 `FastWAM.build_inputs`

输入 batch：

```text
video:        [B,3,T,H,W]
action:       [B,A,Da]
proprio:      [B,A,Dp]
context:      [B,128,4096]
context_mask: [B,128]
```

检查条件：

- `H` 和 `W` 必须是 16 的倍数。
- `T % 4 == 1`，当前 `T=9`。
- `A % (T - 1) == 0`，当前 `32 % 8 == 0`。

VAE encode：

```text
input_latents = VAE.encode(video)
input_latents: [B,48,Tz,Hz,Wz]
Tz = (T - 1) / 4 + 1
Hz = H / 16
Wz = W / 16
```

第一帧 latent：

```text
first_frame_latents = input_latents[:,:,0:1]
first_frame_latents: [B,48,1,Hz,Wz]
```

Proprio token：

```text
proprio[:,0,:] -> Linear(Dp,4096) -> [B,1,4096]
context: [B,128,4096] -> [B,129,4096]
context_mask: [B,128] -> [B,129]
```

### 4.2 Flow matching 加噪

视频：

```text
noise_video:     [B,48,Tz,Hz,Wz]
timestep_video:  [B]
latents_noisy:   [B,48,Tz,Hz,Wz]
target_video:    [B,48,Tz,Hz,Wz]
```

动作：

```text
noise_action:    [B,A,Da]
timestep_action: [B]
noisy_action:    [B,A,Da]
target_action:   [B,A,Da]
```

Scheduler 公式来自 `WanContinuousFlowMatchScheduler`：

```text
noisy = (1 - sigma) * sample + sigma * noise
target = noise - sample
```

### 4.3 VideoDiT `pre_dit`

输入：

```text
x:        [B,48,Tz,Hz,Wz]
timestep: [B]
context:  [B,L,4096]
```

Patch embedding：

```text
Conv3d(in=48,out=3072,kernel=[1,2,2],stride=[1,2,2])
grid:   [F=Tz, h=Hz/2, w=Wz/2]
tokens: [B,Sv,3072]
Sv = F * h * w
```

时间调制：

当前配置 `seperated_timestep=true` 且 `fuse_vae_embedding_in_latents=true`，所以第一个 latent frame 的 token timestep 被设为 0，其他 token 使用采样 timestep：

```text
t_mod: [B,Sv,6,3072]
```

文本/proprio cross-attention context：

```text
text_embedding(context): [B,L,3072]
context_mask expand:    [B,Sv,L]
```

RoPE：

```text
freqs: [Sv,1,128]
```

### 4.4 ActionDiT `pre_dit`

输入：

```text
action_tokens: [B,A,Da]
timestep:      [B]
context:       [B,L,4096]
```

Action encoder：

```text
Linear(Da,1024)
tokens: [B,A,1024]
```

时间调制：

```text
t_mod: [B,6,1024]
```

文本/proprio cross-attention context：

```text
text_embedding(context): [B,L,1024]
context_mask expand:    [B,A,L]
```

RoPE：

```text
freqs: [A,1,128]
```

### 4.5 MoT 每层内部尺度

虽然 video token hidden dim 是 3072，action token hidden dim 是 1024，但它们的 attention Q/K/V 都被投影到同一维度：

```text
Dqkv = num_heads * attn_head_dim = 24 * 128 = 3072
```

每层：

```text
video q/k/v:  [B,Sv,3072]
action q/k/v: [B,A,3072]

concat:
q_cat/k_cat/v_cat: [B,Sv+A,3072]

mixed_attention:
mixed: [B,Sv+A,3072]

split:
video mixed slice:  [B,Sv,3072] -> video self_attn.o -> [B,Sv,3072]
action mixed slice: [B,A,3072]  -> action self_attn.o -> [B,A,1024]
```

随后每个 expert 各自做：

```text
residual gate
cross-attention 到 text/proprio context
MLP
```

### 4.6 `post_dit` 与 loss

Video head：

```text
tokens_out["video"]: [B,Sv,3072]
head projection:     [B,Sv,48*1*2*2]
unpatchify:          [B,48,Tz,Hz,Wz]
```

如果第一帧 latent 被固定，loss 前会去掉第一个 latent time step：

```text
pred_video[:, :, 1:]:   [B,48,Tz-1,Hz,Wz]
target_video[:, :, 1:]: [B,48,Tz-1,Hz,Wz]
```

Action head：

```text
tokens_out["action"]: [B,A,1024]
head:                 [B,A,Da]
```

Loss：

```text
video_loss = MSE(pred_video, target_video), 按 pad mask 加权
action_loss = MSE(pred_action, target_action), 按 action_is_pad 加权
loss_total = lambda_video * video_loss + lambda_action * action_loss
```

## 5. 推理尺度

### 5.1 基础 FastWAM `infer_action`

输入：

```text
input_image: [1,3,H,W]
prompt 或 context/context_mask
proprio: [Dp] 或 [1,Dp]
action_horizon A
```

第一帧编码：

```text
first_frame_latents: [1,48,1,Hz,Wz]
video tokens:        [1,tokens_per_frame,3072]
```

动作 latent：

```text
latents_action: [1,A,Da]
```

基础版会 prefill 第一帧 video K/V cache，然后每个 action denoise step 只更新 action：

```text
pred_action:    [1,A,Da]
latents_action: [1,A,Da]
output action:  [A,Da]
```

### 5.2 `infer_joint`

输入同上，但还需要 `num_video_frames=T`。

初始化：

```text
latents_video:  [1,48,Tz,Hz,Wz]
latents_action: [1,A,Da]
first_frame_latents 写入 latents_video[:,:,0:1]
```

每个 denoise step：

```text
_predict_joint_noise
  -> pred_video:  [1,48,Tz,Hz,Wz]
  -> pred_action: [1,A,Da]

scheduler.step 更新 video/action latent
重新固定第一个 video latent
```

输出：

```text
video:  list[PIL.Image], 长度 T
action: [A,Da]
```

### 5.3 Joint 与 IDM 的推理差异

| 变体 | `infer_action` 是否生成完整 video latent | action 能否看未来 video token | 输出视频 |
|---|---:|---:|---:|
| `FastWAM` | 否，只用第一帧 cache | 否 | 否 |
| `FastWAMJoint` | 是，同步去噪 video/action | 是 | `infer_action` 不返回，`infer_joint` 返回 |
| `FastWAMIDM` | 是，先去噪 video，再用 video KV cache 去噪 action | 是 | `infer_joint` 返回，`infer_action` 只取 action |

## 6. 注意力 mask 尺度

MoT attention mask 是二维矩阵：

```text
attention_mask: [Sv + A, Sv + A]
```

基础 FastWAM：

```text
video -> video: 由 video_expert.build_video_to_video_mask 控制
video -> action: false
action -> action: true
action -> video: 仅 first-frame video tokens true
```

FastWAMJoint：

```text
action -> video: 全部 video tokens true
```

FastWAMIDM 训练：

```text
tokens = [noisy_video, cond_video, action]
attention_mask: [Sv + Sv + A, Sv + Sv + A]
action -> cond_video: true
action -> noisy_video: false
```


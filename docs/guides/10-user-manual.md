# Guide 10: 用户手册 —— 训练与解码全流程

> **Audience**: 所有使用本框架的人（owner + 团队成员）。
> **Status**: 2026-08-26 编写，覆盖提交 `5d7aaed`（AMP/划分/val loop）之后的当前状态。
> 旧版 [04-run-training.md](04-run-training.md) / [06-run-decoding.md](06-run-decoding.md)
> 以本手册为准。

---

## 1. 五分钟上手（本地）

```bash
cd neural-decoding-infra
conda activate recon

# ① 假数据全链路 smoke（1 分钟，无 GPU 也行）
python -m recon.cli.train model=meg_model_a data=fake train.smoke=true paths=local

# ② 真实数据 1 epoch（本地 E:/results，单卡 bf16）
python -m recon.cli.train model=meg_model_a data=drdr paths=local \
    train.smoke=true train.max_steps_per_epoch=8 train.amp_dtype=bf16 \
    data.subjects=[1] data.stories=[1,10,11]

# ③ 解码（需要一个 checkpoint，见第 4 节）
python -m recon.cli.decode --checkpoint <ckpt路径> --subject 1 --story 1 --max-chars 50
```

---

## 2. 配置系统总览（Hydra）

### 2.1 三层组合

启动命令 `python -m recon.cli.train` 时，[train.yaml](../../configs/train.yaml)
的 `defaults` 把三个配置组**拼**成一份完整配置：

```yaml
defaults:
  - model: fmri3dcib    # configs/model/<名字>.yaml
  - data: drdr          # configs/data/<名字>.yaml
  - paths: cluster      # configs/paths/<名字>.yaml
  - _self_              # train.yaml 自己的键优先级最高
```

### 2.2 命令行覆盖

任何叶子键都可以在命令行覆盖，语法 `组.子键=值`：

```bash
python -m recon.cli.train model=meg_model_a data=drdr paths=local \
    data.subjects=[1] data.stories=[1,10,11] \
    data.split.method=ratio data.split.val_ratio=0.2 \
    train.epochs=50 train.batch_size=64 train.lr=1e-4 train.amp_dtype=bf16
```

- 列表值用 `[1,2,3]`；字符串直接写；`null` 表示禁用
- **优先级**：命令行 > train.yaml > defaults 组文件
- 每次启动的**完整解析后配置**会打印在日志开头，并存到
  `outputs/<时间戳>/.hydra/config.yaml` —— 出问题先看这份快照

### 2.3 插值

配置里 `${paths.xxx}` 引用其他键（如 `${paths.results_dir}`），组合后统一解析。

---

## 3. 配置字段大全

### 3.1 `paths/<环境>.yaml` —— 三套路径

| 字段 | 含义 | local（开发机） | cluster（集群） |
|---|---|---|---|
| `data_root` | 原始 BIDS 数据根（预留） | `E:/reconstruction` | `/home/test/reconstruction` |
| `processed_root` | **预处理输入数据**根（训练只读这里） | `E:/results` | `/home/test/reconstruction/results` |
| `results_dir` | **训练输出**根（ckpt/元数据，与输入分离） | `./tests/tmp_results` | `/home/test/reconstruction/outputs` |
| `pretrained_root` | 预训练模型根 | `./tests/fake_pretrained` | `/home/test/reconstruction/pretrained` |
| `gpt_path` | GPT-2 中文权重目录 | 本地实际路径 | `${pretrained_root}/gpt2-chinese-cluecorpussmall` |

切换环境只改一个参数：`paths=local` / `paths=cluster` / `paths=ci`。

### 3.2 `train.yaml` 顶层

| 键 | 含义 | 默认 |
|---|---|---|
| `run_id` | 运行标识，进入 ckpt 目录名（`${now:...}` 自动时间戳；也可设 `RUN_ID` 环境变量覆盖） | 时间戳 |

#### `train:` 段 —— 训练超参

| 键 | 含义 | 默认 |
|---|---|---|
| `epochs` | 训练轮数 | 100 |
| `batch_size` | 每卡批大小（DDP 下是 per-rank） | 8 |
| `lr` / `weight_decay` | 学习率 / 权重衰减 | 1e-4 / 0.01 |
| `optimizer` | `adamw` / `adam` / `sgd` | adamw |
| `scheduler` | `cosine` / `step` / `constant` | cosine |
| `warmup_steps` | ⚠️ **预留，未接入** | 100 |
| `grad_clip` | 梯度裁剪阈值（0 关闭） | 1.0 |
| `abort_on_nan` | loss 非有限值时立即中止（保存应急 ckpt） | true |
| `amp_dtype` | `null`(关) / `bf16`(4060 推荐，无需 scaler) / `fp16`(自动 GradScaler)。**PH402/Pascal 无 16 位硬件，必须 null** | null |
| `num_workers` | DataLoader 进程数（Linux 正常 fork；Windows 自动降为 0） | 2 |
| `pin_memory` | 锁页内存 | true |
| `seed` | 随机种子 | 42 |
| `smoke` | 冒烟模式：强制 1 epoch + 每 story 采样数受限 | false |
| `max_steps_per_epoch` | 每 epoch 最大 step 数（smoke 用，null=不限） | null |
| `log_interval` | 每 N step 打日志 | 10 |
| `save_interval` | 每 N epoch 存周期 ckpt | 10 |
| `eval_interval` | 每 N epoch 跑验证（无 val 集则跳过） | 5 |

#### `eval:` / `ckpt:` / `logging:` 段

| 键 | 含义 | 默认 |
|---|---|---|
| `ckpt.dir` | ckpt 目录模板（`${run_id}` 会被 RUN_ID 替换） | `${paths.results_dir}/ckpt/${run_id}` |
| `ckpt.keep_last_n` | 只保留最近 N 个周期 ckpt（`best_val.pt` 不受影响） | 3 |
| `logging.wandb_project` | W&B 项目名 | neural-decoding-infra |
| `logging.wandb_mode` | `online` 直传 / `offline` 写本地待同步 / `disabled`。**环境变量 `WANDB_MODE` 优先** | online |
| `eval.interval` / `eval.metrics` | ⚠️ **预留，未接入**（val 指标固定为 loss/cos/mse） | — |

### 3.3 `data/drdr.yaml`（MEG）与 `data/drdr_fmri.yaml`（fMRI）

| 键 | 含义 | 默认 |
|---|---|---|
| `subjects` | 被试过滤：`null`=全部；`[1]`=只训 sub 1 | null |
| `stories` | story 过滤（显式列表，或 `null`） | null |
| `n_context` | MEG context 窗口大小（>0 用 `_context_{n}.npy` + 对齐位移；0 用普通文件） | 5 |
| `layer` | zstim 的 GPT-2 层 | 10 |
| `weights` | 4 个 delay 的加权系数（对齐旧管线） | `[0.1, 0.7, 0.5, 0.3]` |
| `max_stories` | 扫描 story 编号上限 | 60 |
| `mask_path` | fMRI 脑掩膜 nii（null = `processed_root/mask.nii`） | null |

#### `data.split` —— 训练/验证/测试划分（按 **story** 粒度，防时间泄漏）

| 键 | 含义 |
|---|---|
| `method` | `none`（全量训练，默认）/ `ratio`（同被试内按比例）/ `holdout`（跨被试留出）/ `explicit`（显式列表） |
| `val_ratio` / `test_ratio` | ratio 模式的验证/测试比例（train 自动拿剩余；小数 floor 到 1） |
| `seed` | 划分随机种子（可复现） |
| `test_subjects` | holdout 模式：整体留出的被试（如 `[12]`；LOSO = 换人重跑） |
| `train_stories` / `val_stories` / `test_stories` | explicit 模式：显式 story 列表（复现旧 `[42,12,6]`） |

```bash
# 同被试解码（论文指标一）：sub 1 内 70/15/15
data.split.method=ratio data.split.val_ratio=0.15 data.split.test_ratio=0.15

# 跨被试泛化（论文指标二）：sub 1-11 训练，sub 12 整体留出测试
data.subjects=[1,2,3,4,5,6,7,8,9,10,11,12] data.split.method=holdout data.split.test_subjects=[12]
```

### 3.4 `model/*.yaml`

| 模型 | 字段 | 含义 |
|---|---|---|
| `meg_model_a` | `n_channels` / `n_context` | 输入通道数 306 / context 窗口（必须与 data.n_context 一致） |
| | `embed_dim` / `gru_hidden` / `gru_layers` / `n_heads` | 编码器/GRU/注意力结构 |
| | `semantic_dim` | 输出语义维度（固定 768） |
| `fmri3dcib` | `input_shape` | 3D 体素形状，真实数据 `[91, 109, 91]` |
| | `backbone_dim` / `bottleneck_dim` / `beta` | CNN 输出维 / IB 瓶颈维 / KL 权重 |

⚠️ **配对规则**：`model=meg_model_a` ↔ `data=drdr`；`model=fmri3dcib` ↔ `data=drdr_fmri`。不要混配。

---

## 4. 训练

### 4.1 三种规模

```bash
# A. 单卡（直接跑，无脚本）
python -m recon.cli.train paths=cluster model=meg_model_a data=drdr \
    data.subjects=[1] data.split.method=ratio \
    train.epochs=100 train.batch_size=64 train.amp_dtype=null \
    logging.wandb_mode=offline

# B. 单节点双卡（torchrun 自动注入环境变量）
torchrun --nproc_per_node=2 -m recon.cli.train paths=cluster model=meg_model_a \
    data=drdr data.subjects=[1] train.batch_size=128 train.amp_dtype=null

# C. 多节点（在 mgmt 上执行，见第 6 节脚本）
bash scripts/launch_multi_node.sh model=meg_model_a data=drdr \
    data.subjects=[1] data.split.method=ratio train.epochs=100 train.batch_size=128
```

### 4.2 启动时会发生什么（fail-fast 检查）

训练开始前依次执行：**设备数核对 → 分布式通信往返测试 → 首个 batch 数据契约检查 →
dry-run 前向+反向**，任何一步失败立即报错退出（秒级），不会训到一半才炸。
随后 rank 0 写 `run_metadata.json`。

### 4.3 产物清单

```
${paths.results_dir}/ckpt/<RUN_ID>/
├── checkpoint_epoch_10.pt   # 周期 ckpt（保留最近 keep_last_n 个）
├── best_val.pt              # 验证集 loss 最优的 ckpt（解码用这个）
└── run_metadata.json        # 本次任务全档案：git commit/平台/模型/数据/划分明细/完整配置
logs/…                       # 各 run 的 stdout 日志
wandb/                        # offline 模式的待同步数据
```

### 4.4 监控、中断与恢复

- 训练进度：rank 0 的 tqdm 进度条（loss + grad 实时）；日志 `tail -f logs/xxx.log`
- **Ctrl-C**：当前 batch 结束后存应急 checkpoint 再退出（rank 0）
- W&B 上传（mgmt 有网）：`bash scripts/sync_wandb.sh`
- 断点续训入口：`Trainer.load_checkpoint()` 已就绪，CLI 入口规划中（P2）

---

## 5. 解码与评估

### 5.1 `python -m recon.cli.decode` 参数表

| 参数 | 含义 | 默认 |
|---|---|---|
| `--checkpoint` | 训练 ckpt 路径。**缺省 = 随机初始化模型**（仅管线冒烟，输出无意义，会警告） | 无 |
| `--subject` / `--story` | 解码哪个被试/故事（**必填**） | — |
| `--output` | 输出文本路径 | `decoded/sub{s}_story{t}.txt` |
| `--processed-root` | 预处理数据根 | 平台自动（win=E:/results，linux=集群路径） |
| `--gpt-path` | GPT-2 权重目录 | 平台自动 |
| `--model-name` / `--model-config` | 模型名（无 ckpt 时用）/ 模型配置 yaml 覆盖 | meg_model_a |
| `--n-context` | MEG context 窗口（与训练一致） | 5 |
| `--beam-width` | beam 宽度 | 200 |
| `--lm-mass` | nucleus 采样概率质量 | 0.9 |
| `--sim-ratio` | 脑信号相似度 vs LM 概率的权重 | 0.15 |
| `--select-layer` | 候选特征的 GPT-2 层（与训练 zstim 层一致） | 10 |
| `--max-chars` | 最多解码步数 | 2000 |

```bash
python -m recon.cli.decode \
    --checkpoint /home/test/reconstruction/outputs/ckpt/megA_sub1_0826/best_val.pt \
    --subject 1 --story 25 --max-chars 200 --beam-width 200
# story 号从 run_metadata.json 的 split.test_stories（或 val_stories）里挑
```

### 5.2 `python -m recon.cli.eval` —— 批量评估

```bash
python -m recon.cli.eval \
    --decoded-dir decoded/          # 解码输出目录（*.txt）
    --reference-dir refs/           # 参考文本目录（同名文件）
    --output eval_report.json       # CRR/CER/Top-k 报告
```

---

## 6. 脚本手册

### 6.1 `scripts/smoke.sh` —— 分级冒烟

```bash
bash scripts/smoke.sh --tier=L0    # 假数据全链路（环境装好后的第一测）
bash scripts/smoke.sh --tier=L1    # sub1 真实 MEG，1 epoch + ratio 划分 + val
bash scripts/smoke.sh --tier=L2    # 4 节点 8 卡 DDP
bash scripts/smoke.sh --tier=all   # 全部
# 日志：logs/smoke_tier*.log
```

### 6.2 `scripts/launch_multi_node.sh` —— mgmt 一键多节点

**机制**：ssh 到每个节点，为每张 GPU 起一个进程，注入 DDP 环境变量
（见下表），等待全部结束；Ctrl-C 会 pkill 所有远程进程。

**自定义**：编辑脚本头部配置：

```bash
NODES=(cn3 gn14 gn15 gn16)   # 节点列表
GPUS_PER_NODE=2              # 每节点卡数
MASTER_PORT=29500            # DDP 握手端口（节点间要互通）
```

**DDP 环境变量协议**（自己写启动脚本时照抄）：

| 变量 | 含义 | 来源 |
|---|---|---|
| `RANK` | 全局进程编号 0..7（节点 i 第 g 卡 = i*2+g） | 脚本 |
| `LOCAL_RANK` | 节点内编号 0/1 → 决定用哪张 GPU | 脚本 |
| `WORLD_SIZE` | 总进程数 = 节点数 × 每节点卡数 | 脚本 |
| `MASTER_ADDR` / `MASTER_PORT` | 握手地址/端口 | 脚本（master 节点 IP） |
| `RUN_ID` | 统一运行名（ckpt/wandb 目录一致） | 脚本 |
| `WANDB_MODE` | `offline`（计算节点无网） | 脚本默认 |
| `NCCL_IB_DISABLE` | `1`（无 InfiniBand） | 脚本默认 |
| `NCCL_SOCKET_IFNAME` | 万兆网卡名（多网卡时设） | 按需 |

torchrun 会自动注入 RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*，所以**单节点多卡直接 torchrun，只有跨节点才需要这个脚本**。

### 6.3 `scripts/sync_wandb.sh` —— W&B 离线同步（mgmt 上）

```bash
# mgmt 节点（有网）：wandb login 一次后
bash scripts/sync_wandb.sh              # 同步 ./wandb 下全部离线 run
bash scripts/sync_wandb.sh <其他目录>   # 指定目录
```

---

## 7. 排错速查

| 现象 | 查 |
|---|---|
| 配置不认识/报错 | `outputs/<时间戳>/.hydra/config.yaml` 看解析后配置；字段名见第 3 节 |
| 启动即退出 | 启动检查日志（设备/通信/数据/dry-run 是哪步报的错） |
| 数据文件找不到 | `paths.processed_root` 与第 1 步侦察结果核对；`discover_drdr` 只认成对的 resp+stim 文件 |
| 形状不匹配 | dry-run 会拦住；检查 `model.n_channels/n_context` 与 `data.n_context` 一致 |
| loss 变 NaN | `abort_on_nan` 已中止并留应急 ckpt；降 lr / 检查数据 |
| DDP 卡死 | 10 分钟超时后报错；查 29500 端口互通、`NCCL_SOCKET_IFNAME` |
| GPU 训不动 | PH402 必须 `train.amp_dtype=null` |
| 详细清单 | [08-debug-checklist.md](08-debug-checklist.md)、[09-manual-review-checklist.md](09-manual-review-checklist.md) |

# Guide 09: 人工审查与集群验证清单

> **Audience**: Owner（单 owner 操作模式，见 ADR-0005）。
> **Status**: 2026-08-20 编写，对应提交 `d42e7e4` / `c097027` / `302d1c2`。

本清单用于两件事：① owner 对 Sprint 1（P0+P1）代码的人工审查；
② owner 在集群上的逐级验证。已自动验证的部分（107 个测试、
本地 CPU/GPU smoke）在此不再重复，只列**必须人眼确认**或**必须集群实测**的点。

---

## Part 1 — 人工代码审查清单

### 1.1 移植保真度（最高优先级）

新模型的"改进"哪些是刻意的、哪些是移植失误，只有人眼能判断：

- [ ] **MEGModelA vs 旧 `MEGReconstructionModel`**：逐层对照
  [recon/models/meg/meg_model_a.py](../../recon/models/meg/meg_model_a.py)
  与 `2026-7-22/model/MEG_model_A.py`。新实现用了 `MultiheadAttention`
  + `ResidualProject`；旧实现的具体结构（是否有相同的 spatial/temporal
  attention、GRU 层数、dropout）需要确认。**若结构不同，模型无法加载
  旧 checkpoint，也无法复现旧结果**。
- [ ] **FMRI3DCIBModel**：对照旧 `fMRI3dCIBModel.py`。已知差异：
  新 backbone 用了 `AdaptiveMaxPool3d(1)` 收尾（旧版是固定输出 1000 维）。
  确认 loss 权重一致：`0.7*cos + 0.3*mse + beta*kl`，`beta=1e-3`（旧 Model1）。
- [ ] **weight_delays 数值等价性**：跑一遍
  `recon/data/drdr.py:weight_delays` 与旧 `datasetbuilder.py:process_stim_data`
  对同一数组的对比，应逐元素相等（float64 计算后 cast float32）。
- [ ] **MEG 对齐位移**：`recon/data/datasets/meg.py` 的
  `j = t - (n_context - 1)` 零填充逻辑，与旧 `load_story_data` 的
  prepend-4-zeros 行为一致（有测试锁定，但语义是否符合预期需要 owner 确认：
  这等价于"用 t-4 时刻的脑响应预测 t 时刻的刺激"）。

### 1.2 解码语义

- [ ] [recon/decoders/beam.py](../../recon/decoders/beam.py) 的文档中列出的
  **刻意简化**（无 WR 模型、1 char/step 对齐、`logprob + sim_ratio*cos`
  组合得分）是否符合下一阶段目标；WR 模型移植排期（P2）。
- [ ] `sim_ratio=0.15` 与旧配置的 `ratio=0.15` 语义不同（旧版是 nucleus
  ratio，新版是 sim 权重）——确认这个超参含义变化是否可接受。
- [ ] 冷启动字符集 `_COLD_START_CHARS` 是否需要换成 GPT 的 [CLS] 起始逻辑。

### 1.3 安全与隐私（每轮提交前必查）

- [ ] `git status` 无数据文件（`*.npy` / `*.pt` / `*.fif` / `*.nii*`）
- [ ] 无 `.env` / API key（`.gitignore` 已覆盖，人工抽查）
- [ ] 无真实 story 文本、无参与者隐私信息
- [ ] LICENSE = Proprietary 未变

### 1.4 依赖与配置

- [ ] [pyproject.toml](../../pyproject.toml) `torch>=2.0` 是通用下限；
  实际版本由环境安装决定（本地 2.6.0+cu124 已验证）。
- [ ] `configs/paths/cluster.yaml` 三个路径与集群实际目录核对
  （`data_root` / `processed_root` / `results_dir` / `pretrained_root`）。

---

## Part 2 — 集群逐级验证步骤

> 全程在 mgmt 节点操作。每级通过后再进下一级。
> 集群 torch 版本先实测再定（见 3.2）。

### T0 · 环境与数据就位

```bash
# 1. mgmt 节点：拉代码
cd /home/test/reconstruction
git clone https://github.com/gaigebell/neural-decoding-infra.git   # 或 rsync 本地副本
cd neural-decoding-infra

# 2. conda 环境（见 standards/07）：合并旧 env 或新建
conda create -n recon python=3.10 -y
conda activate recon
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[dev]"     # 先最小集，[all] 里的 brainomni 等按需再装

# 3. 实测 torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#    期望：2.6.0+cu118 True 'Tesla ...'（PH402 显示名可能是 P40/PH402）
#    cu118 起不来（glibc）→ 换 cu124 试；都不行 → 检查驱动版本

# 4. 数据目录核对
ls /home/test/reconstruction/results/MEG/zresp/zresp1_1.npy
ls /home/test/reconstruction/results/zresp/cube/zresp1_1.npy
ls /home/test/reconstruction/pretrained/gpt2-chinese-cluecorpussmall/config.json
```

**验收**：`torch.cuda.is_available()==True`，三个路径存在。

### T1 · 单节点单卡（真实数据 1 epoch）

```bash
# MEG
python -m recon.cli.train paths=cluster model=meg_model_a data=drdr \
    data.subjects=[1] data.stories=[1,10,11] train.epochs=1 train.save_interval=1 \
    train.amp=false train.batch_size=64
# fMRI
python -m recon.cli.train paths=cluster model=fmri3dcib data=drdr_fmri \
    data.subjects=[1] data.stories=[1] train.epochs=1 train.save_interval=1 \
    train.amp=false train.batch_size=4
```

**验收**：
- [ ] loss 数值合理且与本地同一配置下接近（本地 MEG loss≈1.47/fMRI≈0.77）
- [ ] checkpoint 落在 `results/ckpt/<时间戳>/`
- [ ] `train.amp=true` 再跑一次对比（PH402 无 fp16 tensor core，
      预期更慢或持平 → 正式训练关 AMP）

### T2 · 单节点双卡 DDP

```bash
# 在某个计算节点（如 cn3）本地两卡：
torchrun --nproc_per_node=2 --master_port=29500 -m recon.cli.train \
    paths=cluster model=meg_model_a data=drdr data.subjects=[1] \
    train.epochs=1 train.batch_size=64 train.amp=false
```

**验收**：
- [ ] 两卡 loss 与 T1 单卡一致（数据并行应几乎相同）
- [ ] 无 NCCL 报错（节点内 PCIe 直连，无 NVLink，通信慢但应可用）

### T3 · 多节点 DDP（4 节点 8 卡）

```bash
bash scripts/launch_multi_node.sh model=meg_model_a data=drdr \
    data.subjects=[1] train.epochs=1 train.batch_size=128 train.amp=false
```

**验收**：
- [ ] 8 个 rank 全部启动、无 rank 卡死（脚本内 ssh 每节点 2 进程）
- [ ] loss 与单卡一致
- [ ] 记录 epoch 时间，算扩展性：2 卡 vs 8 卡。无 IB 走万兆 TCP，
      预期 sub-linear，但数据并行对通信不敏感，应该接近线性
- [ ] 若卡死：检查 NCCL 环境变量（见 3.4）与 29500 端口互通

### T4 · 集群解码（训练后）

```bash
python -m recon.cli.decode \
    --checkpoint results/ckpt/<run>/checkpoint_epoch_N.pt \
    --subject 1 --story 1 --processed-root /home/test/reconstruction/results \
    --gpt-path /home/test/reconstruction/pretrained/gpt2-chinese-cluecorpussmall \
    --max-chars 100 --beam-width 200
```

**验收**：
- [ ] 输出为连续中文字符、无 [UNK] 洪泛
- [ ] 若旧管线的解码结果还在（`decode_result/*.txt`），用
      `recon.cli.eval` 对同一 story 对比新旧 CRR——**这是新旧管线
      对齐的最终判据**

### T5 · 断点续训

- [ ] 训练中途 Ctrl-C → 用 `Trainer.load_checkpoint` 恢复 → loss 连续
- [ ] `keep_last_n=3` 清理逻辑正常

---

## Part 3 — 关键检查点提醒（易踩坑汇总）

1. **当前 push 卡在本地网络**：`git push` 到 GitHub 被 connection reset
   （2026-08-20）。本地 3 个提交安全。可选方案：配代理重试 /
   在集群 Windows 管理机上 push / rsync 代码到集群。
2. **集群 torch 版本**：CentOS 7 (glibc 2.17) 大概率只能 cu118
   （最后版本 2.6.0）；cu124 起不来就回 cu118。**先实测再装全家桶**。
3. **无 InfiniBand**：多节点 DDP 走 TCP。必要时
   `export NCCL_IB_DISABLE=1`；万兆网卡名不一致时设
   `NCCL_SOCKET_IFNAME=<网卡名>`。
4. **PH402 = Pascal**：无 fp16 tensor core、无 bf16 → `train.amp=false`。
5. **Windows 专用 guard 不影响 Linux**：`num_workers` 自动降级只在
   win32 触发；集群上 `num_workers=2` 正常 fork。
6. **W&B**：rank 0 需要 `wandb login`；离线集群可设
   `WANDB_MODE=offline`（`WandBLogger` 已支持禁用降级）。
7. **NFS IO**：fMRI cube 单 story ~2GB，首次 mmap 读会慢；训练前可
   预读热身（`dd` 或跑一个 epoch 空转）。后续 P2 考虑预加载到节点本地盘。
8. **端口**：`MASTER_PORT=29500` 需在节点间互通（手动 ssh 模式没有
   slurm 的防火墙豁免，可能要开端口）。
9. **run_id 时间戳**：同一分钟内多次启动会共用 ckpt 目录——正式跑用
   `RUN_ID=<名字>` 环境变量区分（`save_checkpoint` 已支持）。
10. **旧代码参照**：所有"对不对"的疑问以 `2026-7-22/` 旧代码为唯一
    参照系；发现新代码与旧代码不一致时，先判断是刻意改进还是 bug，
    改进项在 Part 1.1 逐条打勾确认。

---

## 已完成（无需重复检查）

- [x] 本地 CPU：107 测试 + MEG/fMRI 真实数据训练 + 解码全链路
- [x] 本地单卡 GPU（RTX 4060）：训练 AMP 修复、解码、回归全绿
- [x] 数据发现、延迟加权、对齐位移、collate 均有测试锁定
- [x] DDP 后端选择：`nccl`（cuda）/`gloo`（cpu）已实现
- [x] `scripts/launch_multi_node.sh` 多节点启动脚本已就绪

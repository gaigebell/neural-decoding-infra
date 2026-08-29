# 下一阶段计划：预处理管线重构与集群部署

> **Status**: 2026-08-27 v2（owner 已确认 5 个问题 + 设计讨论定稿，见
> [data-pipeline-design.md](data-pipeline-design.md)）。目标：预处理管线
> 整合进框架 → 集群部署 → 全被试预处理 → 全被试 MEG model A 训练。

---

## 1. 旧管线盘点（已读代码）

| 组件 | 文件 | 作用 | 状态 |
|---|---|---|---|
| MEG 特征处理器 | `process_brain_feature.py`（714 行） | fif → zresp：306 通道选择(12:318)、z-score、0.4s 窗口/100Hz 采样 | 三个类（BrainFeatureProcessor / MEGFeatureProcessor / BrainOmniFeatureProcessor），路径硬编码 |
| BrainOmni 路径 | `prepipeline.py` | 12 被试循环跑 `brainomni_process(sr=256, seg=2)` | 依赖 `BrainOmni.brainomni.model` 包 + 两个 checkpoint（`BrainTokenizer.pt`/`BrainOmni.pt`/`model_cfg.json`）——**权重位置待确认** |
| 语义特征对齐 | `process_semantic_feature.py` | wordvectors → lanczos 降采样（12s 首字、0.4s 网格）→ 4 delays → zstim | 依赖 `E:/results/wordvectors/story*_{layer}.npy` |
| **GPT-2 字符特征提取** | ✅ 已找回：`brainread/v0/embed.py`（GPT2LMHeadModel 全层 0-12） | 生成 wordvectors | 待重构进框架 |
| fMRI cube | ✅ 已找回：`brainread/v0/dataproc/fMRI/getzresp.py`（nibabel → linear/cube） | nii → (T, 91, 109, 91) | 待重构进框架 |
| BrainOmni | ✅ `D:/allforwork/Liu_Lab/_Reconstruction/BrainOmni-main/`（代码 + `ckpt_collection/` 权重均在本地） | 编码器路径 | 作为 extra 接线 |
| 杂项 | `main_datagen.py`、`extract_semantic_feature.py`（BERT 旧路径） | 编排/旧特征 | english 剔除字典在 3 个文件里重复 |

**旧管线的系统性缺陷**（重构要解决的）：
1. 无断点续跑——12 被试 × 60 story 跑一半挂了从头来
2. 无并行编排——纯 for 循环，单进程
3. 路径硬编码（`E:/results`）、环境隐式（`./preprocessed_data` 相对路径）
4. 无产物校验——生成的对错靠眼睛看
5. 无 provenance——跑完不知道参数是什么（和训练侧的 run_metadata 形成鲜明对比）
6. GPT-2 特征提取、fMRI cube 两个环节代码缺失

## 2. 数据现状

- 原始 BIDS：`mydata/`（sub-01..12，`.datalad` 元数据在），MEG 每 story 一个
  `sub-XX_task-RDR_run-N_meg.fif` + events/channels/coordsystem；fMRI MNI
  在 `derivatives/preprocessed_data/sub-XX/MNI/`
- 时间对齐：`derivatives/annotations/time_align/char-level/*.mat`
- 已有产物：`results/`（sub-1 的 zresp/zstim/cube；**其他被试缺失**）
- GPT-2：`pretrained/gpt2-chinese-cluecorpussmall` ✓；BrainOmni 权重位置未确认

## 3. 业界方案调研（初步）

| 层 | 业界做法 | 对我们的适配 |
|---|---|---|
| 数据标准 | **BIDS + BIDS-Derivatives**；MNE-BIDS 读写 | 原始数据已是 BIDS ✓，产物按 BIDS-derivatives 命名即可 |
| fMRI 预处理 | fMRIPrep / HCP pipelines（DRDR 自带 HCPpipelines_pku 代码） | 我们只需 nii → cube 提取段，沿用旧逻辑 |
| MEG 预处理 | MNE：滤波/降采样/epoching | 旧代码已用 MNE 读 fif；重构时用 mne 公开 API 收敛 |
| 管线编排 | **Snakemake / Nextflow**（DAG + 断点续跑 + 并行）；轻量替代：GNU parallel / joblib | 集群无 slurm → 用 Snakemake 的本地执行器 + ssh 分发，或自研轻量 runner（对齐 launch_multi_node.sh 的手动 ssh 模式） |
| 数据版本 | Datalad（mydata 已带 `.datalad`）、git-annex | 保留现状即可 |
| 产物缓存/校验 | joblib hash / Zarr + 每个产物存 (config, hash) provenance | 每个 npy 旁写 `_meta.json`（参数 + 输入 hash），与训练侧 run_metadata.json 风格一致 |
| 特征提取 | 冻结预训练模型批量推理（transformers） | GPT-2 层 10 特征：补写缺失脚本 |

## 4. 计划（四阶段）

### Phase 1：重构核心处理器（本地，E 盘数据验证）

- 新建 `recon/preprocessing/` 包：`meg.py`（fif→zresp 经典路径）、
  `semantic.py`（GPT-2 特征 + lanczos 对齐 + delays）、`fmri.py`（nii→cube）
- **契约复用 `recon/data/schema.py`**：处理器输出 schema-valid 的
  zstim/zresp 数组，训练侧直接可用
- 每个处理器：输入/输出路径参数化（hydra 配置）、单 story 可跑
- **数值等价性验收**：新管线产物 vs `E:/results` 现有产物逐元素 diff
  （sub-1 已有基线可用！）
- 找回/重写缺失环节：GPT-2 词特征提取（几十行 transformers）；定位
  fMRI cube 生成逻辑；确认 BrainOmni 权重与依赖（brainomni 路径作为
  extra 单独接线，不进主线）

### Phase 2：集群部署与断点续跑

- `configs/preprocess.yaml` + `recon/cli/preprocess.py`（按 subject/story
  粒度、`--resume` 跳过已有产物、每产物写 `_meta.json`）
- 并行：12 被试 × 60 story 用 launch 脚本同款手动 ssh 分发，或
  Snakemake 本地执行器 + ssh（先评估，倾向自研轻量 runner——依赖最少）
- 产物目录：`processed_root` 下按 subject 组织，与训练侧 discover 对齐

### Phase 3：全量预处理 + 全被试训练

1. 12 被试全部 zresp/zstim 生成（预计单被试 ~1-2h 计算 + IO 占大头）
2. 校验：每产物 hash + meta 落盘；抽查 sub-1 与旧产物 diff
3. **全被试 MEG model A 训练**：`data.subjects=[1..12]` + holdout
   （LOSO 循环留出）→ 得到跨被试泛化基线

### Phase 4（后续）：监控完善

- 预处理进度面板（复用训练监控的 NFS 日志方案）
- val 曲线已在 wandb 修复（`ec1a69c`）；补 grad/GPU 利用率面板

## 5. Owner 确认结果（2026-08-27）

1. ✅ BrainOmni 在 `BrainOmni-main/`（本地，含权重）→ Phase 1 作为 extra 接线
2. ✅ fMRI cube 脚本在 `brainread/v0/dataproc/fMRI/`，GPT-2 特征在
   `brainread/v0/embed.py` → 均待重构
3. ✅ 编排 = **自研轻量 runner**（文件系统状态机 + ssh 分发，与训练侧
   launch 脚本同模式；规则数据化保留 Snakemake 升级路径）——理由见设计文档
4. ✅ 集群原始 fif 已齐全（12 被试），无需补传
5. ✅ 格式 = BIDS-derivatives 风格 + 产物 sidecar（`*.meta.json`：输入
   hash + 参数 + 代码 commit）；zresp 按编码器分层（`classic/`、
   `brainomni_v1/`）；E:/results 只读保留作黄金标准

**设计细节全在 [data-pipeline-design.md](data-pipeline-design.md)**：
三层分离（编排/格式/计算）、业界方案对比、预处理边界（重预处理继承
BIDS、特征提取进管线、编码缓存落盘）、BrainOmni token 缓存利弊、
论文阅读清单。

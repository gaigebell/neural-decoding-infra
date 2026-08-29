# 数据管线设计讨论：架构、格式与预处理边界

> **Status**: 2026-08-27，回应 owner 的三个问题：管线架构选型、目标特征清单、
> 预处理边界。本文是设计讨论记录；行动计划见
> [preprocessing-phase.md](preprocessing-phase.md)。

---

## 1. 先把三个概念拆开（大多数选型困惑来自不拆）

| 层 | 回答的问题 | 代表方案 |
|---|---|---|
| **编排** | 谁、在什么时候、以什么顺序、什么并行度跑什么 | 手写脚本 / 自研 runner / Snakemake / Nextflow |
| **存储格式** | 产物怎么组织在磁盘上，下游怎么读 | LeRobot 格式 / BIDS-derivatives / parquet / WebDataset / Zarr |
| **计算** | 每个步骤做什么转换 | 预处理 / 特征提取 / 编码 |

**关键洞察：LeRobot 解决的是格式层，不是编排层。** LeRobot = 每 episode 一个
目录（`meta/episode_XXX.parquet` + 视频文件），数据集自带结构、加载器统一。
它的"管线"就是 codebase 里的普通脚本。所以"想要 LeRobot 格式"（Q2a）与
"选哪个编排工具"（Q3）是**两个正交的决策**，分开选。

## 2. 业界方案图谱

### 2.1 编排层

| 方案 | 工作方式 | 优点 | 缺点 |
|---|---|---|---|
| 手写脚本 + for 循环（旧代码） | 顺序执行 | 零依赖 | 无断点/无并行/无 provenance，挂一半全重来 |
| **自研轻量 runner**（我们训练侧 launch 脚本就是这个路线） | ssh 分发 + 文件系统做状态机 | 依赖最少、完全贴合"无 slurm 集群" | resume/进度/日志要自己写（一次性的小成本） |
| Snakemake / Nextflow | 声明式规则 + 自动依赖图 + 断点续跑 + 并行 | 成熟、生态好、可复现性强 | 学习成本；规则粒度要设计；CentOS 7 上装 conda/mamba 集成是额外运维负担；Nextflow 偏容器化 |
| 工作流平台（Airflow/Dagster/Prefect） | 服务化调度 | 企业级监控 | 运维重量级，研究项目杀鸡用牛刀 |

**判断标准不是"功能多少"，而是"依赖图长什么样"**：如果任务之间互相依赖
（A→B→C 有分支合并），DAG 引擎赢；如果任务是**完全同构的 embarrassingly
parallel**（我们的场景：12 被试 × 60 story 的独立任务），for 循环 + 断点
文件 + ssh 分发 = 3 行 shell 的事，Snakemake 的依赖图退化成摆设。

### 2.2 格式层

| 格式 | 结构 | 优点 | 缺点 |
|---|---|---|---|
| **BIDS + derivatives** | `sub-XX/modality/` + sidecar JSON | 神经科学事实标准；审计清晰；多模态原生支持；工具链（mne-bids 等） | 面向原始数据设计，衍生数据约定较松 |
| **LeRobot** | 每 episode 一目录：parquet 表格 + 视频 | 表格（parquet 列式）加载快、可懒读、可增量；"一个目录一个数据集"心智负担低 | 为机器人回放设计，无多模态/被试概念，需要映射 |
| **HuggingFace datasets** | 远程托管 + 分片 parquet/arrow | 生态最大；streaming 免下载 | 依赖 HF hub；中国网络不可靠；私有数据不合规 |
| **WebDataset / MDS** | tar/分片 + 索引 | 大训练吞吐、随机访问 | 格式门槛高，研究阶段过重 |
| **Zarr** | 分块压缩数组 | 超大数组、云原生 | 工具链少，我们数据量（GB 级）用不上 |

### 2.3 我们的选择（结合约束）

约束：单 owner 运维、无 slurm、CentOS 7、NFS 共享存储、研究迭代速度优先、
未来接新数据集（多模态）。

**格式 → BIDS-derivatives 风格 + 产物 sidecar**（吸收 LeRobot 的精神）：
- 目录按 `processed_root/{modality}/{sub-XX}/` 组织（BIDS 血统，审计清晰）
- **每个产物一个 sidecar**：`zresp_1_10.npy` 旁放 `zresp_1_10.meta.json`
  （输入文件 hash + 全部参数 + 代码 git commit + 生成时间）——这就是
  LeRobot 的"结构化可审计"落到神经数据上的样子
- 多模态扩展 = 新写一个 adapter 产出同一结构（c）

**编排 → 自研轻量 runner**（与训练侧 launch 脚本同一套模式）：
- 任务清单（subject×story×modality）落一个 JSON；worker 扫描"无 sidecar
  或 hash 不匹配"的任务并认领 → **文件系统就是任务队列**，天然断点续跑、
  崩溃安全、可多机并行
- 保留升级路径：runner 的规则数据结构化（输入/输出/命令），将来想换
  Snakemake 只改生成层

## 3. 预处理边界（Q3，最重要的一层）

### 3.1 总原则

**管线做"确定性的、跑一次就够的、昂贵的"转换；训练循环做"频繁迭代的、
可随机的"处理。** 每把一层处理往上游固化，下游迭代就快一步；但每缓存
一层，就要多存一份数据 + 多管一层 provenance。平衡点：**缓存到训练直接
可读的粒度**，中间态按需重算。

### 3.2 脑电预处理做多少？（你不熟悉，正好说清）

分层看：

| 层 | 内容 | 谁来做 |
|---|---|---|
| **重预处理** | 滤波、去伪迹（ICA/SSP）、坏道插值、重参考、降采样 | **DRDR 官方 BIDS derivatives 已做**——继承，不重做 |
| **特征提取** | 通道选择（12:318）、z-score、时间窗、lanczos 对齐 | **我们管线的活**：确定性、便宜、跑一次缓存 |
| **编码** | BrainOmni token 化 | 缓存推理结果（见 3.4） |

- 重预处理是专业性最强、最耗时、最难复现的一层——业界（fMRIPrep 之于
  fMRI、MNE 之于 MEG）的做法是把它固化成**带版本号的一次性流程**，产物
  入库，下游不再碰原始数据。我们数据是官方预处理过的，所以 Phase 1
  根本不用碰 MNE 的滤波/ICA；未来接新数据集（原始 fif）时，再按 MNE
  标准流程写独立阶段。
- 语言侧同理：BIDS 时间对齐已把字符 onset/offset 对齐好，**没有传统
  NLP 清洗的活**（不用分词、不用去停用词）；我们文本"预处理"= 读对齐 +
  GPT-2 特征 + lanczos 重采样。`eliminate_data`（英语剔除）这类数据集
  特例进 config，不进代码。

### 3.3 存储账（回答"缓存合不合算"）

sub-1 全量产物体量：zresp 60 story × ~1011×306×4B ≈ **74 MB**；zstim
60 × 1011×768×4B ≈ **186 MB**；fMRI cube 47 story × 570×91³×4B ≈
**75 GB**（大头，但已有）；GPT-2 wordvectors ≈ 几十 MB。**除了 fMRI
cube，其余都是"该缓存"的量级**——换来的是一次训练少跑一遍编码器。

### 3.4 BrainOmni token 落盘——对，但要分层命名

先回答"对不对"：**对**。这是业界多模态对齐工作流的标准做法（CLIP 特征
缓存、语音 fbank 缓存、LLM embedding 缓存——全部都是把编码器推理结果
落盘，让对齐阶段只做对齐）：

- **利**：对齐头结构/loss 每次实验迭代不用重跑编码器（BrainOmni 前向
  往往比对齐头训练还贵）；token 跨实验复用；训练 dataloader 读 npy 极快
- **弊**：编码器选择被固化——换 BrainOmni 版本/权重/segment 长度 → 重跑
  提取（所以要版本化命名 + sidecar 记权重 hash）；多一份磁盘 + provenance
- **旧代码的真实问题**：`zresp/` 目录里混着"经典路径"和"brainomni 路径"
  两种产物（甚至 sr100/segment2 等变体），靠文件名靠猜 → 新格式
  **按编码器分层**：`zresp/classic/`、`zresp/brainomni_v1/`，sidecar 记
  权重 hash + 采样率 + segment 长度

## 4. 论文/资料阅读清单（按主题，按需读）

**神经数据标准与预处理**
- Gorgolewski et al. 2016, "The brain imaging data structure" (BIDS) — 数据组织圣经
- Esteban et al. 2019, "fMRIPrep: a robust preprocessing pipeline" (Nature Methods) — 预处理工程化的范本
- Gramfort et al. 2013, "MEG and EEG data analysis with MNE-Python" — MEG 处理基线
- 本项目数据集论文：Wang et al., DRDR（OpenNeuro ds004078 的 Scientific Data 论文）

**解码任务定位（决定管线要产出什么）**
- Tang et al. 2023, "Semantic reconstruction of continuous language from non-invasive brain recordings" (Nature Neuroscience) — 语义对齐范式
- Chen et al. 2023, "A high-performance speech neuroprosthesis" 及 Willett et al. 2023 (Nature) — 特征提取+对齐的工作流参照
- BrainOmni（如公开，arXiv）——你们的编码器，读它的数据约定

**数据管线工程**
- Cadene et al., "LeRobot" (2024, arXiv) — 数据集格式+管线组织
- Lhoest et al. 2021, "Datasets: A Community Library for NLP" — HF datasets 设计
- WebDataset（PyTorch 官方文档）——分片格式思想
- Köster & Rahmann 2012, "Snakemake" (Bioinformatics) — 编排引擎原理（选了 runner 也要懂它的取舍）

**可复现性**
- Wilkinson et al. 2016, "The FAIR Guiding Principles" (Scientific Data)
- DVC 官方文档的 "Data versioning" 部分

## 5. 对行动计划的影响（已并入 preprocessing-phase.md）

1. 缺失环节已全部找回：GPT-2 特征（`brainread/v0/embed.py`）、fMRI cube
   （`brainread/v0/dataproc/fMRI/getzresp.py`）、BrainOmni 代码+权重
   （`BrainOmni-main/`，含 `ckpt_collection/`）
2. 集群原始 fif 已齐全（12 被试），无需补传
3. 编排定为自研 runner；格式定为 BIDS-derivatives + sidecar；zresp 按
   编码器分层命名
4. 等价性验收照旧：E:/results 产物只读保留作黄金标准

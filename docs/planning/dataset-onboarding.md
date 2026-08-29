# 新数据集接入流程（草案）

> **Status**: 2026-08-27 起草。**本文是暂定流程，尚未经过真实新数据集接入的
> 实践核验**——未来做数据扩展时，按实际经验修订、案例化并提升为正式指南
> （届时取代或并入 guides/02-write-data-adapter.md）。
> 背景设计见 [data-pipeline-design.md](data-pipeline-design.md)。

---

## 心智模型

**正常情况（新数据集、同范式）**：adapter 3 个函数 + runner 跑 + 训练侧
1 行配置 = 一个数据集接入完成。DRDR 自身是第一个"客户"，Phase 1 会顺带
把这条路走通。

## 第 0 步：盘点与决策（人工，约半天）

对照预处理三层边界（重预处理 / 特征提取 / 编码），做三个判断：

| 判断 | 分支 |
|---|---|
| 数据是**原始**还是**已预处理**？ | 原始 → 补一个"重预处理阶段"（MNE 滤波/ICA/坏道插值，一次性跑、产物入库、带版本号——fMRIPrep 模式）。已预处理 → 跳过（DRDR 的情况） |
| **时间对齐**怎么给？ | char-level onset/offset → 直接复用 lanczos 对齐（DRDR）。word-level → 先对齐到字符。**没有对齐（自由讲述）→ ASR + 强制对齐**，这是最重的分支 |
| **语义编码器**用哪个？ | GPT-2 某层（默认）/ BERT / BrainOmni → 决定特征维度与 zstim 约定 |

## 第 1 步：写 adapter（代码层，核心工作）

在 `recon/preprocessing/` 下加 `dataset_x.py`，实现 3 个函数：

```
discover()            → 扫描新数据集的目录 → (subject, story) 索引
extract_brain(s, t)   → 数据集特有读取 → 统一 zresp 形状（进 schema 契约）
extract_semantic(s,t) → 文本/刺激读取 → 统一 zstim（延迟加权 768）
```

**关键设计**：adapter 只写"该数据集特有"的部分（文件格式、通道命名、
时间戳解析）。以下为**共享基础设施，adapter 零代码获得**：
lanczos 对齐、延迟加权、z-score 规范化、sidecar 写入（输入 hash + 参数 +
git commit）、schema 校验。

## 第 2 步：跑转换（零新代码）

任务清单（subject × story × 阶段）交给 runner → 并行、断点续跑、
sidecar 校验自动生效。产物落在统一目录：
`processed_root/{dataset}/{modality}/{sub-XX}/`。

## 第 3 步：训练侧接入（约一行配置）

统一格式 ⇒ 训练侧几乎不动：`discover_drdr` 泛化为
`discover_dataset(name)`；训练命令 `data=<dataset_name>` + 现有 split
配置照用。跨数据集混合训练时，subject 加数据集前缀（`x_sub-01`）防冲突。

## 第 4 步：验收

- schema 契约自动挡形状/维度错误
- 与新数据集原始文件抽查比对
- 若原作者发布过特征产物（如官方 zresp）→ **数值等价性 diff**（与
  DRDR 拿 E:/results 当黄金标准同一套路）

---

## 已知会打破此流程的三种情况（诚实边界）

1. **新模态**（EEG/ECoG/fNIRS）：格式层无影响（schema 加 sample 类型 +
   collate 分支），但训练侧要有对应模型——管线不背这个锅
2. **无时间对齐的文本**：重点在对齐层（ASR/强制对齐），特征提取反而是
   小事
3. **非连续叙事范式**（trial-based：图像刺激、P300）：时间轴语义完全不同，
   "4 延迟加权"约定可能不适用 → 需为新范式定义新的对齐约定，adapter
   接口要扩一个"对齐策略"维度

## 未来核验清单（实践时逐条回答）

- [ ] 第 0 步的三个判断是否覆盖了实际遇到的全部情况？有没有第 4 种分支？
- [ ] 3 个函数的接口签名是否够用（脑/语义之外是否需要独立的时间轴函数）？
- [ ] "共享基础设施"清单是否真的零改动就能复用？
- [ ] subject 前缀方案在跨数据集训练时是否引起其他冲突（story 命名、split）？
- [ ] 重预处理阶段的产物管理（该缓存在哪一层）是否符合预期？

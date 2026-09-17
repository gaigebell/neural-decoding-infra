# 解码与评估子系统：方案讨论与计划 v2

> **Status**: ✅ 2026-09-17 方案已批准，进入实现。owner 三项确认：
> ① eval 默认双口径都报、flag 指定指标；② 语义相似度用 BERT（与 legacy
> 的 SentenceBERT 口径一致）；③ legacy 解码产物在本地 `recon/result/`
> （story 60，MEG_model_A + baseline，含 BLEU/Levenshtein/Meteor/
> SentenceBERT 对比 CSV）。
>
> 前置条件：跨被试泛化（LOSO sub-12）与跨被试故事泛化两实验已完成训练。

---

## 1. 现状盘点

**已有**：BeamSearchDecoder（真 beam + GPT-2 layer-10 特征 + 单字符词表 +
batched brain encoding）、decode.py（单 story，仅 MEG）、eval.py（CRR/CER/
top-k/perplexity）。

**缺**：批量评估入口、参考文本构建、fMRI 解码、BrainOmni tokens 解码、
新旧管线 CRR 对齐判据。

## 2. 决策点（v2）

### D1 · 解码-训练对齐：双轨方案（已定）

**问题回顾**：训练是 `resp_ctx[t-4] → stim[t]`；decode.py 现状用
`resp_ctx[t]`（偏移 4 步）；1 char/step 只走 1011 步 vs 真实 1586 字符。

**v2 决议**：
- **轨道 A（主，完成度要求高）**：直接映射。字符 onset → zresp 行
  `row=(onset-12)/0.4`，模型输入取 `resp_ctx[row-4]`；**解码长度 =
  字符数（1586）**——等价于"告知模型要预测多少个字，让模型填满"。
  纯函数、可测试、与训练完全同构。
- **轨道 B（预留接口，不实现）**：`LengthPredictor` 抽象（脑信号 →
  预期字数），为将来 WR 模型评估保留插入点。设计上要求 A 的解码循环
  接受"外部长度"参数，B 实现后替换即可。

### D2 · 批量评估入口 + 双口径（已定，含新增问题）

```bash
python -m recon.cli.decode_batch --checkpoint <best_val.pt> --split test
python -m recon.cli.eval --decoded-dir ... --reference-dir ... [--chinese-only]
```

- test 集自动发现：run_metadata 的 split 明细（holdout 留被试 / explicit
  留故事两种形态都支持）
- 参考文本：`.mat` 重建 + legacy 同款 regex（数字分组）+ eliminate_data 剔除
- **双口径**（owner 提出的中文评估问题）：
  - **全字符口径**：含标点数字——与 legacy CRR 对齐（D7 判据用它）
  - **纯中文口径**（`--chinese-only`）：过滤非中文后重算 CRR/CER——
    标点在中文里位置自由，剔除后更能反映语义恢复质量
  - 建议：eval 默认同时报告两个口径

### D3 · 解码范式全谱 + 通用架构（重点扩充）

**范式调研**（脑解码文献 + LLM 生成技术两侧）：

| 范式 | 原理 | 适用场景 | 现状 |
|---|---|---|---|
| **Beam search** | 束搜索 + 脑语义评分 | 当前主线；质量/速度平衡 | ✅ 已有 |
| Greedy | beam=1 特例 | 极速冒烟/基线 | beam 特例 |
| **流式解码** | 按时间窗边收边解，逐段输出 | 实时脑机接口演示、长故事分段 | ❌ 缺 |
| **Prompt/条件生成** | 脑预测语义作软提示/前缀注入大 LM | 换大 LM 后；利用预训练语言先验 | ❌ 缺 |
| 检索式 | 预测向量 → 最近邻句子库 | 封闭集合任务 | ❌ 不计划 |
| seq2seq 条件生成 | 脑特征 cross-attention → 文本 | 需专门训练，异于现有对齐范式 | 远期 |

**通用架构**（在 beam 之上抽象，保证可扩展）：

```
解码系统 = 可插拔组件组合
├── FeatureSource（特征来源）
│     MEG 语义 | fMRI 语义 | BrainOmni tokens(+对齐头)
│     接口: story 信号 → per-char 预测向量序列
├── Scorer（评分器）
│     接口: (候选字符上下文, 脑预测向量) → (logprob, sim)
│     默认: GPT-2 层 10 特征（可换大 LM）
├── GenerationStrategy（生成策略）
│     BeamSearch | Greedy | Streaming | PromptConditioned
│     接口: (scorer, features, length) → 文本
└── LengthPredictor（字数估计，D1-B 预留）
      接口: 脑信号 → 预期字数
```

**本阶段落地顺序**：BeamSearch 完整化（对齐修复 + 长度参数化）→
FeatureSource 接口（fMRI 顺手接入）→ 其余策略留接口不实现。
**BrainOmni 对齐头**：训练任务独立排期（不阻塞本阶段）。

### D4 · 参考文本 tokenization（不变）

legacy 同款 regex（`[一-鿿]|\d+\.\d+|\d+|[^一-鿿\w\s]`，数字分组为多字符
token）+ eliminate_data 剔除后与 zstim 行对齐。

### D5 · 效率：先 profiling 后优化 + 教程（新增教程交付物）

原则不变（先测后改，不预设 KV cache）。**新增交付物**：
[guides/11-llm-inference-optimization.md](../guides/11-llm-inference-optimization.md)
——LLM 推理优化入门教程（owner 无相关经验）：prefill/decode 两阶段、
KV cache 原理与显存账、为什么我们的场景 KV cache 收益有限、batch/
量化/speculative decoding/continuous batching 概念、torch.profiler 实操、
Pascal 特别说明、我们的现状对照与"何时做哪步"决策表。

### D6 · 评测指标：四视角调研（重点扩充）

**计算机/NLP 视角**：CER/WER（编辑距离族）、BLEU-1/chrF（字符级
n-gram）、BERTScore（embedding 相似度）、perplexity（语言流畅度）✅部分已有
**心理学视角**：语义保真度（decoded 与 ref 的 embedding 余弦——衡量"意思
对了几分"，最贴近心理语言学对语义恢复的定义）
**语言学视角**：信息完整性（长度比、唯一字率——粗粒度但零成本）、
n-gram 重叠（可选）
**神经科学/脑解码文献视角**：Tang 2023 等用 WER + LLM-embedding 语义
相似度；中文脑解码（legacy）用 CRR（位置字符正确率）

**推荐组合（第一版评估报告）**：

| 层 | 指标 | 理由 |
|---|---|---|
| 主口径 | CRR + CER（双口径：全字符 / 纯中文） | legacy 对齐 + 语义纯净 |
| 语义层 | embedding 余弦相似度（GPT-2 层 10 平均池化） | 心理语言学最相关 |
| 流畅度 | perplexity（decoded 在 LM 下） | 已有 |
| 完整性 | 长度比 + 唯一字率 | 零成本诊断 |

分层报告：per subject / per story / per split；**legacy 对齐口径必报**。

### D7 · 新旧管线判据（已确认可做）

owner 确认 legacy `decode_result/*.txt` 还在 → 同一 story 新旧管线解码 →
CRR 对比作为迁移最终验收，写入评估报告。

## 3. 优先级（v2）

| 优先级 | 内容 |
|---|---|
| **P0** | D1-A 对齐修复（含长度参数化）+ D2 批量评估（双口径）+ D4 参考文本 |
| P0.5 | 两实验评估验收 + D7 新旧对比（若 legacy 产物就位） |
| **P1** | D3 架构重构（FeatureSource/Scorer 解耦 + fMRI 输入） |
| **P2** | D5 profiling 与优化 + D6 语义/完整性指标 + 教程跟进更新 |
| 预留 | LengthPredictor（D1-B）、BrainOmni 对齐头、流式/Prompt 策略 |

## 4. Owner 决策记录（2026-09-17，全部确认）

1. ✅ D2：eval **默认双口径都报**，`--metrics` flag 指定子集
2. ✅ D6：语义相似度用 **BERT**（bert-base-chinese；与 legacy 的
   SentenceBERT 同族口径）
3. ✅ D7：legacy 产物在本地 `recon/result/`
   （`1_test_60_MEG_model_A_0.9_0.1_8_200.txt` = 200 条 beam 假设 +
   `compare_results_story60.csv`：BLEU/Levenshtein/Meteor/SentenceBERT）。
   判据执行：故事泛化实验（test 集含 story 60）解码 story 60 → 新旧
   CRR/CER/sem_sim 对比。

> P0 已实现（`decode_batch` / `eval` 双口径 / 对齐修复），P0.5 进行中。

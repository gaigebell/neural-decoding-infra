# Guide 11: LLM 推理优化入门（为脑解码量身定做）

> **Audience**: 无 LLM 推理优化经验的 owner；也供团队成员学习。
> **Status**: 2026-09-14 编写，配合
> [planning/decode-eval-phase.md](../planning/decode-eval-phase.md) D5。

本文用我们自己的场景（GPT-2 中文、beam search 解码、PH402 集群）讲清
"LLM 推理慢在哪、怎么优化、我们该做哪些"。概念优先，公式最少。

---

## 1. 自回归生成的两阶段

LLM 生成是**逐 token 循环**：每次输入"已有文本"，输出"下一个 token"。
每轮循环分两段：

| 阶段 | 做什么 | 成本 |
|---|---|---|
| **prefill** | 处理输入序列（已有文本）的所有 token，一次前向 | 与序列长度成正比 |
| **decode** | 基于上一步缓存，只算新 token 一个位置 | 恒定、便宜 |

**关键洞察**：如果不做任何缓存，第 t 步会重新计算前 t-1 个 token 的全部
注意力——总成本 O(n²)。KV cache 就是让 decode 阶段只算新 token。

## 2. KV cache 原理与显存账

Transformer 的注意力：`Q · K^T → softmax → × V`。第 t 步生成时：

- 已有 token 的 K/V **不变**（因果注意力只看过去）→ 存起来复用
- 每步只算新 token 的 Q/K/V，拼上缓存 → 每步 O(n) 而非 O(n²)

**显存账**（GPT-2 12 层 / 768 维 / 12 头为例）：

```
每 token 的 KV = 2(每层 K+V) × 12 层 × 768 维 × 2 字节(fp16) ≈ 36 KB
1000 token 序列 ≈ 36 MB（可忽略）；大模型 × 长序列才是问题
```

## 3. 为什么我们的 beam 里 KV cache 收益有限

我们的解码有特殊性：

- **上下文恒定 5-6 个字符**（`context_words=5`）——prefill 本来就只有
  6 token，重算成本≈0。KV cache 省的是"长前缀重复计算"，我们没有长前缀
- 我们**每步的候选特征提取已经批量**（beam×extensions 个 6-token 序列
  一次前向）——这已经是 batching 优化
- **真实瓶颈**（本地实测 GPU 只比 CPU 快 1.5×）大概率是：每步大量
  **小批量前向的启动开销**（kernel launch、CPU-GPU 同步、python 循环），
  而非矩阵计算本身

**结论**：先 profiling 找真凶，别急着上 KV cache。教程第 5 节教你怎么测。

## 4. 优化手段全谱（概念 + 对我们的适用性）

| 手段 | 原理 | 对我们的适用性 |
|---|---|---|
| **KV cache** | 见上 | 🟡 收益有限（短上下文），但换大 LM/长 prompt 后必需 |
| **Batching** | 多个序列打包一次前向，摊薄启动开销 | ✅ **已做**（候选特征批量）；可深化（跨 beam 合并） |
| **量化** (int8/int4) | 权重低精度，省显存省带宽 | ❌ PH402 无 fp16/int8 tensor core，量化反而可能更慢；4060 上可试 |
| **算子融合** | 多个小 kernel 合成一个（如 flash-attention 的融合注意力） | ❌ flash-attn 需要 Ampere+；Pascal 不可用 |
| **Continuous batching**（vLLM 概念） | 服务场景：动态打包多个请求的 decode 步 | ❌ 我们是离线批量解码，不是在线服务 |
| **Speculative decoding** | 小模型草稿、大模型验证，一次验证多个 token | 🟡 脑解码的评分器特殊（脑语义+LM 联合），未必适用；概念值得知道 |
| **torch.compile** | 图优化/融合 | 🟡 Pascal 上支持有限；本地 4060 可试 |

## 5. Profiling 实操（先测后改）

**第一步：时间分解**（不用工具，先在代码里打点）：

```python
t0 = time.perf_counter(); ...; print(f"propose: {t-t0:.3f}s")   # LM 前向
t0 = time.perf_counter(); ...; print(f"features: {t-t0:.3f}s")  # 候选特征前向
t0 = time.perf_counter(); ...; print(f"scoring: {t-t0:.3f}s")   # 余弦+beam 更新
```

我们的 `_nucleus_propose` / `_candidate_features` / beam 更新三段各占多少？
哪段占比最大就优化哪段。

**第二步：torch.profiler**（看 kernel 级细节）：

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    decoder.decode(brain_tensor)          # 跑一小段（如 10 步）
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

关注：`cuda_time_total` 最大的 kernel、CPU 与 CUDA 时间比（若 CPU >> CUDA
说明瓶颈在 python/启动开销，batching 收益大；反之在算力）。

## 6. Pascal（PH402）特别说明

- **无 fp16 tensor core**：fp16 算力 1/64 速率 → AMP 已关；量化收益同样受限
- **无 bf16**；flash-attention / torch.compile 支持有限
- **结论**：集群上的解码优化主要靠**算法侧**（减少前向次数、合并批量），
  硬件侧空间小；本地 4060（Ampere 系，有 bf16）可做硬件侧实验

## 7. 现状对照与"何时做哪步"

| 我们已做 | 对应手段 |
|---|---|
| batched brain encoding（整 story 一次） | Batching ✅ |
| 候选特征批量前向（beam×ext 一次） | Batching ✅ |
| 上下文窗口恒定 5-6 token | KV cache 收益有限的原因 |
| 单字符词表限制（7369/21128） | 搜索空间裁剪 ✅ |

**决策表**：

| 触发条件 | 该做的优化 |
|---|---|
| P0 后 profiling 显示 python 循环占比高 | 深化 batching / 减少每步同步 |
| 换大中文 LM（亿级参数） | KV cache 必须；考虑 vLLM |
| 需要在线服务（实时演示） | continuous batching + 流式架构（D3） |
| 换 Ampere+ 集群 | 重测 fp16/bf16、flash-attn、torch.compile |

## 8. 延伸阅读

- HuggingFace 官方文档 "LLM Inference" 章节
- vLLM 论文（Kwon et al. 2023）——continuous batching 概念出处
- FlashAttention 论文（Dao et al. 2022）——算子融合思想
- 我们的 profiler 实测结果出来后，回填本文第 5 节的真实数据

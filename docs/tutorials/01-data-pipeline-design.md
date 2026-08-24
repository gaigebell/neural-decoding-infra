# Tutorial 01 — 数据管线设计：从假设出发，逐步打破假设

> **Status**: Stable (teaches method, not current code state)
> **Audience**: 想掌握数据管线设计思维的人
> **Last updated**: 2026-08-20
> **Prerequisites**: 会用 Python/numpy；读过
> [`architecture/02-data-pipeline.md`](../architecture/02-data-pipeline.md)
> 了解本项目的管线现状

---

## 0. 怎么读这篇教程

这篇教程模仿 Andrej Karpathy 的教学方式：**不从定义出发，从假设出发**。
每一阶段先立一个"世界是怎样的"假设，写出在这个假设下最自然的代码，
然后用真实数字证明下一个规模下这个假设如何崩塌，最后给出当时的决策和
技术。读完你会掌握五件事：

1. **设计思路**：数据管线的本质是一个"假设 + 契约"的层级结构
2. **设计方法**：加载单元怎么选、边界在哪里画
3. **编写技巧**：惰性加载、缓存、预取的惯用写法
4. **分析方法**：四本账（内存/存储/读取/性能）怎么算、怎么测
5. **改进思维**：瓶颈定位五步法、什么时候该提前设计、什么时候该 YAGNI

贯穿案例是我们项目真实的数据：**MEG 样本 6 KB/条，fMRI 体素样本
3.4 MB/条**——一个天上一个地下，恰好能让每个规模的问题都现出原形。

---

## Stage 0 · 假设："数据全部装得进内存"

### 世界模型

数据几百 MB，一台机器，写代码的人只想赶紧把模型训起来。最自然的写法：

```python
# v1: eager pipeline（本项目旧代码 datasetbuilderMEG.py 就是这种）
def build_dataset(story_list, config):
    dataset = []
    for story in story_list:
        zresp = np.load(f"{config['resp_path']}/zresp1_{story}.npy")
        zstim = np.load(f"{config['stim_path']}/sub1_zstim_10_story{story}.npy")
        for t in range(len(zstim)):                     # 逐行拆成样本
            dataset.append((torch.from_numpy(zresp[t]),  # 脑信号
                            torch.from_numpy(zstim[t]))) # 目标
    return dataset                                      # List[(Tensor, Tensor)]
```

优点：直观、一行一个样本、零抽象。缺点在 Stage 1 才会显形。

### 这个阶段就该学会的分析：内存账

无论规模多小，**先算账再写代码**。内存账公式：

```
常驻内存 = 样本数 × 单样本字节数 × (脑信号 + 目标)
```

算一下我们的 MEG（subject 1 全部 60 个 story）：

```
单样本 zresp  (5, 306)  float32 → 6,120 B
单样本 zstim  (3072,)   float64 → 24,576 B
story 1 有 1011 个时间步 → 60 stories ≈ 60,000 样本
常驻内存 ≈ 60,000 × 30 KB ≈ 1.8 GB   ← 一个 subject
12 subjects ≈ 22 GB                  ← 全部数据
```

**22 GB 超出 65 GB 机器吗？** 没有。但加进模型、优化器、梯度之后——
v1 在我们项目里真实的下场是：跨 subject 实验（旧代码
`dataset_build_meg_cross_sub`）跑 3 个 subject 就开始 OOM 和 30 分钟
的启动时间。**假设"装得进内存"没死，只是打响了丧钟。**

### 这个阶段就该养成的习惯

- 每个数据文件旁边记下 shape 和 dtype（我们就记在
  `architecture/02-data-pipeline.md` §3）
- 启动时打印数据集大小和形状（v1 有这个好习惯）
- **把"加载"和"使用"写成两个函数**——哪怕现在直接调用，这个缝是
  未来所有改动的接入口

---

## Stage 1 · 假设："数据装得进磁盘，但装不进内存"

### 世界模型

数据 10–100 GB（我们：fMRI cube 一个 subject 就 **97 GB**）。
v1 直接物理死亡：`np.load` 那一刻就是 OOM。

### 决策一：不再"加载"，改为"发现 + 按需读取"

```python
# v2: 发现阶段只建索引，不读数据
def discover(data_root, modality):
    index = []                                   # DRDRIndex
    for sub in range(1, 13):
        for story in range(1, 61):
            if resp_exists(sub, story) and stim_exists(sub, story):
                index.append((sub, story))       # 几十 KB 的内存代价
    return index
```

**为什么成对才收录**：v1 的世界里"文件缺失"是加载时的 Warning +
`return None`，下游每处都要防 None；v2 把缺陷移到发现阶段，之后所有
代码假设索引是干净的。实测证据：fMRI 的 story 56–60 有 cube 无
zstim，52 个里只有 47 个成对。**缺陷越早发现越便宜。**

### 决策二：选择加载单元（granularity）

不能逐样本 load（一次 I/O 只读 6 KB 太碎），也不能整数据集 load
（回到 v1）。中间粒度有哪些候选？

| 单元 | 大小（fMRI） | 问题 |
|---|---|---|
| 整数据集 | 97 GB | v1 死法 |
| 单 story | 2.06 GB | 一次读入 2 GB 还是太贵 |
| 单样本 | 3.4 MB | 文件句柄爆炸、I/O 碎片化 |
| **mmap story** | **0（映射）+ 按页读取** | ✅ |

**mmap 的本质**：`np.load(f, mmap_mode="r")` 不复制任何字节，只是告诉
OS"这个文件映射到我的地址空间"。真正的读取发生在访问某一行时，OS
按 4 KB 页从磁盘 fault 进来，热页留在页缓存里自动淘汰。

于是"加载单元"这个经典难题被拆成两层：**管理单元 = story**（每 story
一个 memmap 句柄，缓存进 dict）；**读取单元 = 页**（OS 决定，4 KB）。

```python
# v2: 惰性 dataset
class StoryDataset(Dataset):
    def __init__(self, index):
        self._items = []                       # (sub, story, t) 扁平表
        self._cache = {}                       # (sub,story) -> memmap
    def _story(self, sub, story):
        if (sub, story) not in self._cache:
            self._cache[(sub, story)] = np.load(path(sub, story), mmap_mode="r")
        return self._cache[(sub, story)]
    def __getitem__(self, i):
        sub, story, t = self._items[i]
        return self._story(sub, story)[t]      # 一行，一次页故障
```

**内存账重新计算**（3 个 story 的 MEG 训练）：

```
常驻：3 × (加权 stim 缓存 3.1 MB) ≈ 9 MB   ← 与 story 数成正比
其余：页缓存（OS 淘汰）                    ← 与"最近读过什么"成正比
数据集大小 97 GB：无所谓
```

**决策三：把贵的事做一次**。stim 的延迟加权
（`(T,3072) @ weights → (T,768)`）如果逐样本算，每个样本都要做一次
4×768 的矩阵乘；按 story 算一次并缓存，`__getitem__` 里只剩行切片。
同时完成 16× 压缩（3072 float64 → 768 float32）。

### 这个阶段的分析工具

- **单样本读取账**：`bytes = prod(shape) × dtype.itemsize`。MEG 6 KB
  一行内连续；fMRI 3.4 MB ≈ 830 页连续——连续性是页缓存效率的核心
- **常驻内存账**：`缓存数 × 单元大小 + 批大小 × 样本大小`
- **冷启动时间**：mmap 数据集构建时间 ≈ 0（只建表）

---

## Stage 2 · 假设："IO 不是瓶颈"

### 世界模型

上了 GPU，训练循环变成：读 batch → 移显存 → 前向 → 反向。MEG 一个
batch 在 GPU 上 20 ms，但读 batch + 预处理要 80 ms。**GPU 利用率 20%，
显卡在等磁盘。**

### 分析：吞吐方程

```
每批时间 ≈ max(IO 时间, 计算时间)      # 谁慢谁做主
IO 时间  = 单批字节数 / 有效带宽 + 每样本开销 × 批大小
```

逐项算账：磁盘 SATA ~550 MB/s；NFS 万兆 ~1.1 GB/s 带宽但**每次 4 KB
页故障是一次 RPC 往返（0.2–1 ms）**——小随机读时延迟主导，带宽数字
是骗人的。

### 技术一：预取与流水线（num_workers + pin_memory）

`DataLoader(num_workers=2, pin_memory=True)` 的本质：worker 进程
用 CPU 时间**提前**把未来几个 batch 读好、搬到锁页内存，主进程在
GPU 计算时"顺手"取下一个 batch。IO 被藏进计算的水线里。

**平台差异必须显式化**：Linux fork 直接可用；Windows spawn 会
死锁（我们的 trainer 实测 hang 过一次，现在自动降级并打印警告）。
**"在哪个平台跑"是数据管线的第一公民假设。**

### 技术二：shuffle 与局部性的权衡（最容易被忽略的一课）

`shuffle=True` 打乱的是**全局**顺序：相邻 batch 会随机跳到不同
story → 页缓存全 miss → Stage 1 精心设计的 mmap 优势被自己人击穿。

三个选项，从易到难：

| 方案 | 缓存友好 | 随机性 | 适用 |
|---|---|---|---|
| 全量 shuffle | ❌ | ✅ 完美 | 数据小（MEG） |
| story 块内 shuffle（自定义 sampler） | ✅ | ⚠️ 块间顺序固定 | 中型数据 |
| 预取/预热 + 全量 shuffle | ✅ | ✅ | 大数据的正确答案 |

我们当前的取舍：MEG 规模下无感，全量 shuffle 保留；fMRI 大规模训练
前必须升级（已记入 review checklist）。

### 这个阶段的验证方法

不要猜，测：同一份数据跑 50 batch，`torch.utils.bottleneck` 或直接
计时 IO 段；`nvidia-smi` 看 GPU util 从 20% → 90% 就是赢了。**每次
改进后复测并记录数字**——没有数字的优化是玄学。

---

## Stage 3 · 假设："一台机器"

### 世界模型

8 卡 DDP，数据在 NFS 上，每张卡的进程各自 `__getitem__`。三个新问题：

1. **N 个进程 × 各自页故障**：同一页被 8 个进程各 fault 一次
   （页缓存是共享的——同一台机器没事；**跨节点 NFS 不共享**，每节点
   各读一份）
2. **NFS 延迟 × 碎片化**：Stage 2 的 RPC 延迟被 8 路放大
3. **每 rank 不能看到全量数据**：`DistributedSampler` 把索引分片，
   每 rank 只处理自己的 1/8

### 决策与预算

```
NFS 上的总 IO 预算 = 带宽 1.1 GB/s ÷ (每个 batch 字节 × batches/s)
8 卡同时跑 MEG：8 × 400 KB × 50 batches/s ≈ 160 MB/s   ← 够
8 卡同时跑 fMRI：8 × 14 MB × 5 batches/s ≈ 560 MB/s    ← 接近上限
```

结论（也是我们 checklist 里写的）：**MEG 直接跑；fMRI 多节点前必须
做数据放置**——预热到每节点本地盘，或让每节点只持有它要读的 story
分片。这就是"数据放置"决策：什么数据、在什么时间、离计算多近。

---

## Stage 4 · 假设："数据静态、单项目、规模已知"

### 世界模型

飞轮转起来之后：数据会更新、会被别的项目复用、要复现一年前的实验。

这个阶段的方向（**我们还没做，只在这里立路标**）：

1. **数据版本化**：预处理产物带 hash/manifest——"这份 97 GB 的
   cube 是 2026-08 版还是 2026-09 版？"现在只能靠目录名信仰
2. **流式格式**（WebDataset / tar shard）：把数百万小样本打成大
   分片，小文件问题、NFS 元数据风暴、随机读碎片化一次解决——
   Stage 1 的 mmap 方案在"文件数×进程数"过千时也要让位
3. **异步 IO**：读取与预处理流水线化（`dataloader2` / 自研），
   GPU 永远不空等
4. **采样统计**：每个 epoch 实际读了哪些样本、分布如何——数据
   管线的可观测性和代码一样重要

**但今天不做。** 这是本教程最后一条也是最重要的一条：**每个阶段只
解决当前规模的问题，但要给下一阶段留缝**（我们的缝就是
`discover → Dataset → collate` 的接口边界——换 WebDataset 只需要换
Dataset 实现，trainer 一行不动）。

---

## 工具箱：分析方法和改进思维

### 四本账（每次写数据代码前先算）

```
存储账：总量 = Σ 文件数 × shape × itemsize         （磁盘）
内存账：常驻 = Σ 缓存单元大小 × 缓存数 + 流水线在途  （RAM）
读取账：单样本字节 = prod(shape) × itemsize；
        连续性 = 样本在文件内的布局（行连续 > 跨文件）
性能账：每批时间 = max(IO, 计算)；IO = 字节/带宽 + 延迟 × 故障数
```

### 怎么测（按可信度排序）

1. **微基准**：`time.time()` 包住 `__getitem__` 循环；`/usr/bin/time -v`
   看 max RSS（常驻内存的真实数字）
2. **端到端**：50 batch 训练计时；`nvidia-smi` GPU util
3. **系统级**：`vmtouch` 看页缓存命中；`iostat` 看磁盘繁忙度
4. **profiler**：`torch.utils.bottleneck` / `torch.profiler`

### 改进思维五步法

```
① 量化现状（四本账 + 测量，写下数字）
② 定位瓶颈（吞吐方程指出 max 哪边赢）
③ 提出假说（"页缓存 miss 是主因"）
④ 最小改动（先做 30 分钟能做完的那个方案）
⑤ 复测对比（数字说话；无效就回滚——这是最小改动的意义）
```

### 贯穿全教程的四个设计原则（本项目 ADR 的哲学）

1. **缺陷前移**：能在发现阶段拒绝的，不留到加载阶段；能在边界
   校验的（pydantic），不留到训练中途
2. **契约在边界**：`discover → Dataset → collate` 每层边界都有明确
   的形状契约，坏数据在进入下一层前就死
3. **贵的事做一次**：按单元缓存、按 batch 摊销
4. **平台差异显式化**：Windows/Linux、本地/NFS、磁盘/网络——假设
   写进代码（guard + warning），而不是靠使用者的信仰

---

## 映射表：本教程的技术在本仓库的落点

| 教程阶段 | 技术 | 仓库位置 |
|---|---|---|
| Stage 1 发现 | 成对收录、缺陷前移 | `recon/data/drdr.py::discover_drdr` |
| Stage 1 惰性 | memmap + story 缓存 + 扁平样本表 | `recon/data/datasets/meg.py`, `fmri.py` |
| Stage 1 一次做 | 延迟加权缓存 | `drdr.py::weight_delays` + `_story_stim` |
| Stage 1 契约 | pydantic schema 边界校验 | `recon/data/schema.py` |
| Stage 2 流水线 | num_workers + pin_memory + Windows guard | `recon/engine/trainer.py::build_dataloaders` |
| Stage 2 shuffle | 全量 shuffle（当前取舍）+ 升级路线 | checklist §3.1 |
| Stage 3 分片 | DistributedSampler | `trainer.py::_build_drdr_*` |
| Stage 3 放置 | NFS 预热（P2 路标） | checklist §3.7 |
| Stage 4 路标 | 版本化/流式/异步（未实现） | 本教程 §Stage 4 |

---

## 延伸阅读

- 现状详解：[`architecture/02-data-pipeline.md`](../architecture/02-data-pipeline.md)
- 写新 adapter：[`guides/02-write-data-adapter.md`](../guides/02-write-data-adapter.md)
- 测试分层（含真实数据 Tier 1）：[`standards/04-testing.md`](../standards/04-testing.md)

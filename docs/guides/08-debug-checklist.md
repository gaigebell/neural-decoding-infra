# Guide 08: Debug checklist

> **Audience**: Anyone hitting a bug. Read this before opening a long debugging session.
> **Status**: 2026-09-12 更新——加入集群实战坑位。

---

## Philosophy

> **80% of bugs are configuration errors.** Before debugging code, check config.

## Tier 0: 集群实战坑位（按出现频率排序）

| 症状 | 根因 | 修复 |
|---|---|---|
| 跨节点 `No route to host` (errno 113) | MASTER_ADDR 取了管理网 IP（`hostname -I` 第一个） | 用**主机名**（/etc/hosts 解析到万兆网）或 `MASTER_ADDR=10.0.1.x` |
| `LexerNoViableAltException: N`（数字是 rank） | 脚本 heredoc 里 `$@` 泄漏了 rank/local_rank | build 函数内 `shift 2` 吃掉前两个参数 |
| hydra "config directory .../recon/cli/configs" | CLI 传 `--config-path` 时按调用文件解析 | 删掉（装饰器已指定） |
| DDP 卡死但单 rank 报错 | 一个 rank 崩了，其他等在 all_reduce | 看最早报错的那个 rank 的日志（10 分钟超时兜底） |
| wandb 面板缺 val/lr 曲线 | 同 step 多次 `wandb.log` 丢键 | 每 epoch 合并成一次 log 调用（已修） |
| `CUDA error: invalid configuration argument`（encode） | 全 story 一次前向超出显存 | `encode_chunk` 分块（窗口=整段，分块数值等价） |
| 滤波后数据"全零" | 显示精度问题（fT 量级 %.4f 打不出） | 用 `%.3e` 或看 `std` |
| `ModuleNotFoundError: deepspeed` | BrainOmni 的 vq.py 顶层 import | 推理路径用 no-op 桩（`_ensure_deepspeed_shim`），不装 deepspeed |
| 模型输入报 dtype/尺度错 | 输入尺度 ≠ 训练分布 | brainomni 用 `x_scale=9.508e9`（默认）；其他模型查黄金 |

## Tier 1: Configuration (5 minutes)

Check these first. If any is wrong, fix it and re-run.

- [ ] Are you on the right machine? (`hostname` should be `cn3`, `gn14`, etc.)
- [ ] Are paths correct? `paths.data_root` matches where data actually is.
- [ ] Is `model` the model you think? Check `MODEL_REGISTRY` for valid names.
- [ ] Is `data.subjects` what you intended?
- [ ] Is `paths=cluster` set? (Default might be `local`.)
- [ ] Are env vars set? (`WANDB_API_KEY`, `MASTER_ADDR`, etc.)

```bash
# Print full resolved config without running
python -m recon.cli.train --cfg job
```

## Tier 2: Data (10 minutes)

- [ ] Run the dataset's `__getitem__` directly:

```python
from recon.data.datasets.drdr import DrdrDataset
ds = DrdrDataset(...)
sample = ds[0]
print(sample.brain.x.shape, sample.brain.story_id)
```

- [ ] Are shapes correct? Compare with [Data card](../research/02-data-card.md).
- [ ] Are dtypes correct? (`float32` for tensors, `int` for IDs)
- [ ] Are values in expected range? (e.g., MEG values should be ~[-5, 5] after z-score)
- [ ] Do `story_id` and `subject_id` match across `zresp` and `zstim`?

```python
sample.stim.zstim.shape  # should be (T, 4*768) typically
```

## Tier 3: Model (15 minutes)

- [ ] Forward pass on a single sample (CPU is fine for shape check):

```python
from recon.models.registry import build_model
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/model/fmri3dcib.yaml")
model = build_model(cfg)
x = torch.randn(1, 1, 53, 63, 52)
y = model(x)
print(y.shape)  # should be (1, 768)
```

- [ ] Forward on GPU (if previous step passed):

```python
model = model.cuda()
x = x.cuda()
y = model(x)
```

- [ ] Compute loss on dummy target:

```python
target = torch.randn(1, 768).cuda()
loss, _ = model.compute_loss(y, target)
loss.backward()
```

- [ ] Check `model.parameters()` for NaN/Inf:

```python
for name, p in model.named_parameters():
    if torch.isnan(p).any() or torch.isinf(p).any():
        print(f"NaN/Inf in {name}")
```

## Tier 4: Training loop (30 minutes)

- [ ] Run smoke test on cluster: `train.smoke=true`
- [ ] Check W&B run for first few steps — does loss decrease?
- [ ] If loss is NaN: lower LR (`train.lr=1e-5`), check for unnormalized inputs
- [ ] If loss plateaus immediately: check data (Tier 2)
- [ ] If OOM: lower `train.batch_size`
- [ ] If dataloader hangs: check file paths and permissions

## Tier 5: Multi-node / DDP (1 hour)

- [ ] All ranks reach `sinfo` (oh wait, SLURM is broken) — use `ps` instead

```bash
# On each node
ssh cn3 "ps aux | grep python"
ssh gn14 "ps aux | grep python"
ssh gn15 "ps aux | grep python"
ssh gn16 "ps aux | grep python"
```

- [ ] `MASTER_ADDR` is reachable from all nodes:

```bash
ssh gn14 "curl -v telnet://10.0.1.3:29500 2>&1 | head -5"
```

- [ ] `MASTER_PORT` is not blocked by firewall
- [ ] All nodes use the same git SHA:

```bash
for n in cn3 gn14 gn15 gn16; do
    ssh $n "cd /home/test/reconstruction/neural-decoding-infra && git rev-parse HEAD"
done
```

- [ ] Enable verbose logging:

```bash
NCCL_DEBUG=INFO python -m recon.cli.train ...
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m recon.cli.train ...
```

## Tier 6: Decoding (1 hour)

- [ ] Check checkpoint loads correctly:

```python
ckpt = torch.load("path/to/best.pth")
model.load_state_dict(ckpt["model_state_dict"])
print(ckpt.get("epoch"), ckpt.get("best_metric"))
```

- [ ] Forward on 1 sample gives sensible output:

```python
out = model(brain_sample.x[:1].cuda())
print(out.shape, out.norm(dim=-1))  # last should be ~10-20 typically
```

- [ ] Compute similarity with ground-truth:

```python
gt = stim_sample.zstim[:1].cuda()
sim = torch.cosine_similarity(out, gt)
print(sim)  # should be > 0 for trained model
```

- [ ] If decoding produces garbage: check LM is loaded correctly
- [ ] If decoding is slow: KV cache might not be enabled

## Common error patterns

| Symptom | Likely cause |
|---|---|
| NaN loss after warmup | LR too high; unnormalized inputs; bug in loss |
| Loss not decreasing | Wrong data; wrong target; LR too low |
| OOM | Batch too large; model too large; gradient accumulation issue |
| DDP hangs at init | NCCL can't reach master; wrong MASTER_ADDR |
| DDP slow | No IB → expected for >4 ranks; reduce world_size |
| Decoding produces same chars repeatedly | Beam too narrow; LM prior too strong |
| Decoding produces random chars | Brain encoder output is noise; check checkpoint |

## Log everything

When filing an issue or asking for help, include:

1. Exact command line
2. Resolved config (`python -m recon.cli.train --cfg job`)
3. Git SHA (`git rev-parse HEAD`)
4. Node name + GPU info (`nvidia-smi`)
5. W&B run link
6. Last 30 lines of relevant log

## See also

- [Cluster card](../research/03-cluster-card.md)
- [Guide 04: Run training](04-run-training.md)
- [Guide 05: Launch multi-node](05-launch-multi-node.md)
- [Standard 04: Testing](../standards/04-testing.md)

---

Maintained by owner.
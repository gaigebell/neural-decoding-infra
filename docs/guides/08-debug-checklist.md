# Guide 08: Debug checklist

> **Audience**: Anyone hitting a bug. Read this before opening a long debugging session.

---

## Philosophy

> **80% of bugs are configuration errors.** Before debugging code, check config.

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
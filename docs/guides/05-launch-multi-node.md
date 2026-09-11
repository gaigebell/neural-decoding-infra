# Guide 05: Launch multi-node training

> **Audience**: Anyone running a multi-node DDP experiment.
> **Status**: 2026-09-12 更新——以 `scripts/launch_multi_node.sh` 为准；
> 此前的手动 4-SSH 会话方案已被脚本取代（ADR-0003/0004）。

---

## 节点拓扑（2026-09 实测）

| Node | 计算网 IP（p5p1 万兆） | GPU |
|---|---|---|
| `gn11` | 10.0.1.11 | 2 × PH402 |
| `gn12` | 10.0.1.12 | 2 × PH402 |
| `cn3` | 10.0.1.3 | 2 × PH402 |
| `gn14` | 10.0.1.14 | 2 × PH402 |
| `gn15` | 10.0.1.15 | 2 × PH402 |
| `gn16` | 10.0.1.16 | 2 × PH402 |

共 **6 节点 / 12 GPU**。网络要点（均已设为脚本/trainer 默认，环境变量可覆盖）：
`MASTER_ADDR` 用主机名（/etc/hosts 解析）、`NCCL_SOCKET_IFNAME=p5p1`、
`NCCL_IB_DISABLE=1`。坑：`hostname -I` 第一个 IP 是管理网 → 跨节点
"No route to host"。

## 一键启动（mgmt 上）

```bash
cd /home/test/reconstruction/neural-decoding-infra

# 预览将执行的命令（不真正启动）
DRY_RUN=1 bash scripts/launch_multi_node.sh model=meg_model_a data=drdr

# 正式启动（8/12 卡全量）
RUN_ID="megA_loso12" bash scripts/launch_multi_node.sh \
  model=meg_model_a data=drdr \
  data.subjects=[1,2,3,4,5,6,7,8,9,10,11,12] \
  data.split.method=holdout data.split.test_subjects=[12] \
  train.epochs=100 train.batch_size=128 train.amp_dtype=null \
  train.save_interval=10 train.eval_interval=5
```

脚本行为：ssh 到每节点每 GPU 一个进程（连续 RANK + LOCAL_RANK）、统一
RUN_ID、`WANDB_MODE=offline`、等待全部结束、Ctrl-C 远程 pkill。日志在
mgmt 终端（或重定向到 `logs/`）。

## 常用环境变量覆盖

| 变量 | 默认 | 用途 |
|---|---|---|
| `NODES` | 6 节点 | 换节点列表 |
| `MASTER_ADDR` | 主机名 | 手动指定 master（网络异常排查） |
| `MASTER_PORT` | 29500 | 端口（节点间要互通） |
| `RUN_ID` | 时间戳 | 固定运行名（ckpt/wandb 目录） |
| `CONDA_ENV` | ndinf | 计算节点的 conda 环境 |
| `NCCL_SOCKET_IFNAME` | p5p1 | 万兆网卡名 |

## 验收清单（每轮训练前 30 秒扫一遍）

- [ ] 全部 rank 打印 `DDP initialized: rank=X/12`
- [ ] `Comm check passed: all_reduce round-trip OK`
- [ ] `Dry-run passed`（52/52 params have grads）
- [ ] 首 epoch loss 与单卡基线同量级（MEG ≈1.7）
- [ ] 扩展性：epoch 时间 ≈ 单卡/N × 1.2（8 卡实测 7.3×）

详细排坑见 [Guide 08](08-debug-checklist.md)。

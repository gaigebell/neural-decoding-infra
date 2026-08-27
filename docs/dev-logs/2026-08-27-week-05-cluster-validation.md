# Dev Log 2026-08-27 — 集群验证周：T0→T3 全通过 + 多节点 DDP

## 完成的事

1. **集群环境就位**：cn3 安装 torch 2.6.0+cu124（CentOS 7 实测可用，
   推翻了此前"必须 cu118"的预测——glibc 顾虑不成立）。
2. **T0（fake 数据）/ T1（真实 sub-1 MEG 单卡）通过**：
   60 stories 发现、42/9/9 划分、epoch 63.7s、val 1.66、best_val.pt 落盘。
3. **T3（4 节点 8 卡 DDP）通过**：epoch **8.7s**（7.3× 扩展），
   全 rank 通信检查通过，val/ckpt 正常。
4. **W&B offline** 在计算节点正常工作（rank 0 写 NFS，待 mgmt sync）。

## 踩过的坑（按时间序）

| 坑 | 根因 | 修复 |
|---|---|---|
| hydra "config directory .../recon/cli/configs" | CLI `--config-path` 相对调用文件解析 | 删掉（装饰器已指定） |
| `LexerNoViableAltException: 4` | launch 脚本 heredoc 里 `$@` 泄漏了 rank/local_rank | `shift 2` |
| 跨节点 `No route to host` | `hostname -I` 取到管理网 IP（192.168.0.x） | **MASTER_ADDR 用主机名**（对齐旧管线 `gn11`） |
| （预判坑）NCCL 数据面走错网卡 | 旧管线有 `NCCL_SOCKET_IFNAME=p5p1`，重构时被误删 | 恢复为默认（脚本 + trainer setdefault） |

## 教训

- **旧代码的"魔法数字"往往是验证过的事实**（p5p1、gn11 主机名）——重构时
  删任何默认值前，先 grep 旧代码找依据。这次的教训来自我主观判定
  `p5p1` 是"拍脑袋"，实际是集群实测值。
- 多节点网络的三个事实（p5p1 / 10.0.1.x / 主机名解析）已写入 memory
  的 cluster-config-ph402.md。

## 下一步

- 正式训练：sub-1 MEG model A，100 epochs，8 卡（预计 ~15 分钟）
- mgmt 端 `scripts/sync_wandb.sh` 上传离线 run
- 训练后 decode test stories → CRR/CER 基线
- P2：监控面板、断点续训 CLI 入口

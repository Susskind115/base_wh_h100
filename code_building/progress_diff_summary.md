# Progress Diff Summary

### v0.1.15.12o: BSP-G vLLM SP-Parity 实验

- 在 `codex_coding/src/bench_bsp_moe_dp2.py` 中新增 `G) C12-BSP-G-AttnReduceScatterSP`。
- G 将 attention output projection 从 TP all-reduce/full layout 输出改成 TP reduce-scatter/SP layout 输出。
- G 之后 residual、post-attention norm、MoE 直接消费 SP hidden state，进一步延长 SP layout 生命周期。
- 仍不修改 dInfer/vLLM 源码，不引入 Scheme3 payload 改造，不改变 EB/s_mask 算法。

#### C12 no-quality 两轮结果

| Config | Run1 ms/fwd | Repeat ms/fwd | Avg ms/fwd | Avg vs A |
|---|---:|---:|---:|---:|
| A baseline | 75.35 | 76.70 | 76.03 | - |
| B BSP-MoE | 74.48 | 74.77 | 74.62 | -1.85% |
| C BSP-DelayGather | 73.98 | 75.79 | 74.89 | -1.50% |
| D BSP-DelayGather-M3EPReduce | 74.56 | 80.54 | 77.55 | +2.01% |
| E BSP-CrossLayerSP | 71.67 | 71.59 | 71.63 | -5.78% |
| G BSP-G-AttnReduceScatterSP | 69.55 | 69.51 | 69.53 | -8.55% |
| F BSP-AllReduceFullProbe | 72.28 | 72.34 | 72.31 | -4.90% |

所有 C12 配置 path counts 一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。

#### Component timing 归因

- G 把 `moe.bsp_chunk` 从 E 的 `0.872 ms/fwd`、count `5320` 降到 `0.003 ms/fwd`、count `266`。
- G 新增 `attn.tp_reduce_scatter=5.020 ms/fwd`，`attn_rs_payload=661.502 MB/fwd`。
- G 的 MoE dispatch payload 仍为 `206.719 MB/fwd`，TP gather payload 仍为 `165.375 MB/fwd`，与 E 相同。
- 结论：G 的收益来自 attention output 直接进入 SP layout、减少反复 full/SP chunk，而不是减少 MoE combine/gather 字节。
- component timing 的绝对 e2e 排序受 instrumentation 扰动；机制归因有效，最终速度以 no-quality e2e 为主。

#### Quality smoke

- 小规模 `batch=32,gen=32` snippets 显示 G 没有灾难性语义崩坏。
- 仍需完整质量集才能做生产质量结论。

#### 环境问题

- 第一次 component timing 在 A baseline prefill OOM；原因是旧 benchmark PIDs 报告每卡占用约 `56-58 GB`。
- 清理/释放后重跑成功；该问题记录到 `.learnings/ERRORS.md` 的 `ERR-20260428-001`。

#### 新增关键文件

- codex_coding/results/bsp_moe_bspg_smoke_20260428.json
- codex_coding/results/bsp_moe_bspg_c12_e2e_20260428.json
- codex_coding/results/bsp_moe_bspg_c12_e2e_repeat_20260428.json
- codex_coding/results/bsp_moe_bspg_c12_component_20260428.json
- codex_coding/results/bsp_moe_bspg_quality_smoke_20260428.json
- code_building/process_docs/v0.1-init-project/v0.1.15.12o-bsp_g_vllm_sp_parity.md

#### 本轮命令（追加）

- python3 -m py_compile codex_coding/src/bench_bsp_moe_dp2.py
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --component-timing --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare

### v0.1.15.12p: BSP-G2 vLLM SP-Parity Bundle

- 在 `codex_coding/src/bench_bsp_moe_dp2.py` 中新增 `G2) C12-BSP-G2-SPParityBundle`。
- G2 新增 `SPAttentionInput` carrier，让 attention path 接收 SP-normalized hidden，并在 attention 内部做 input all-gather。
- G2 继续用 attention output reduce-scatter 返回 SP layout，让 residual、post-attention norm、MoE 保持 SP。
- E 和 G 均保持不变；G2 只作为独立 compare config。
- 新增 reduced matrix：
  - `--config-set bspg2`: A/E/G/G2/F
  - `--config-set aeg2f`: A/E/G2/F
  - `--config-set aeggf`: A/E/G/G2/F
- 仍不修改 dInfer/vLLM 源码，不引入 Scheme3 payload 改造，不改变 EB/s_mask 算法。
- 按新规范在实验开始前创建过程文档，并在每个实验阶段结束后立即更新归档。

#### C12 no-quality 结果

| Config | ms/fwd | vs A | Path counts |
|---|---:|---:|---|
| A baseline | 75.428 | - | `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931` |
| E BSP-CrossLayerSP | 71.816 | -4.789% | 同 A |
| G BSP-G-AttnReduceScatterSP | 69.676 | -7.626% | 同 A |
| G2 BSP-G2-SPParityBundle | 69.661 | -7.646% | 同 A |
| F BSP-AllReduceFullProbe | 72.380 | -4.042% | 同 A |

G2 与 G 只差 `0.015 ms/fwd`，属于测量噪声内持平。

#### Component timing 归因

- G2 成功把 attention-input gather 从 MoE/wrapper bucket 迁移到 attention bucket：
  - G: `moe.tp_all_gather=2.662 ms/fwd,count=5054`
  - G2: `attn.input_all_gather=2.573 ms/fwd,count=4788`
  - G2 residual: `moe.tp_all_gather=0.141 ms/fwd,count=266`
- G2 payload 只是迁移而非减少：
  - G: `tp_gather_payload=165.375 MB/fwd`
  - G2: `attn_input_gather_payload=156.671 MB/fwd` + `tp_gather_payload=8.704 MB/fwd`
- G/G2 共同开销仍在：
  - `attn_rs_payload=661.502 MB/fwd`
  - `dispatch_payload=206.719 MB/fwd`
- component instrumentation 下 G2 比 G 慢：
  - G `78.149 ms/fwd`
  - G2 `79.786 ms/fwd`

#### 质量 smoke

- 小规模 `batch=32,gen=32` snippets 显示 G2 与 G 基本一致，未见新增灾难性语义崩坏。
- E 在小 batch 下再次出现 `hot_skip=893`，但 C12 正式 invariant 正常，记录为小 batch artifact。

#### 最终判断

- G 仍是 measured-best 性能路径。
- G2 是 source-organization/SP-parity 路径：attention 同时 owning input gather 和 output reduce-scatter。
- G2 不减少通信字节或同步点，因此不能替代 BSP-H/F2。
- 下一步若继续做性能，应聚焦减少/fuse synchronization 或继续延长 SP layout 生命周期，而不是单纯迁移 gather bucket。

#### 新增关键文件

- codex_coding/results/bsp_moe_bspg2_smoke_20260428.json
- codex_coding/results/bsp_moe_bspg2_c12_e2e_20260428.json
- codex_coding/results/bsp_moe_bspg2_c12_component_20260428.json
- codex_coding/results/bsp_moe_bspg2_quality_smoke_20260428.json
- code_building/process_docs/v0.1-init-project/v0.1.15.12p-bsp_g2_sp_parity_bundle.md

#### 本轮命令（追加）

- python3 -m py_compile codex_coding/src/bench_bsp_moe_dp2.py
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare --config-set bspg2 --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --config-set bspg2 --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --config-set bspg2 --component-timing --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare --config-set bspg2

### v0.1.15.13: EB HetEval512 Law Probe

- 新增观测型脚本 `codex_coding/src/collect_eb_heteval512_laws.py`，在不改变 EB 路由策略的前提下记录压缩后的 `S_mask`、EB top4、no-EB top4/top8、per-layer expert histogram 和 request grouping 摘要。
- 按新规范在实验开始前创建过程文档 `code_building/process_docs/v0.1-init-project/v0.1.15.13-eb_heteval512_law_probe.md`，实验结束后在同一归档补充结果和结论。
- HetEval512/C12 配置确认：
  - `batch=512,gen=256,block=32`
  - `dp=2,tp=4,ep=8`
  - `threshold=0.90`
  - EB: `K=8,topk=4,K_target=40,q_major=1.0,skip_m=5`
- 全量运行 path counts 与历史 C12 一致：
  - `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`
- 主结果：
  - EB top4 unique active experts mean `197.98`
  - no-EB top4 unique active experts mean `251.22`
  - no-EB top8 unique active experts mean `254.59`
  - EB vs no-EB top4 active expert reduction `21.19%`
- 层差异明显：
  - 最强压缩层：L17 `25.17%`, L7 `24.90%`, L9 `24.17%`, L16 `24.02%`
  - 最弱压缩层：L1 `13.73%`, L0 `14.24%`
- `S_mask` 稳定性：
  - adjacent Jaccard mean `0.9862`, p50/p90/p95 为 `1.0`
  - previous-call same-size set coverage `0.9658`
  - previous-block cold coverage `0.9328`
- 控制实验：
  - EB `S_mask` coverage `0.9681`
  - random same-size set coverage `0.7733`
  - offline global-popularity same-size set coverage `0.9704`
- 关键结论：
  - EB 不是随机裁剪，active-expert reduction 真实存在。
  - HetEval512 大 batch 下存在很强的全局热门 expert 结构；后续应探索 `global prior + EB correction`。
  - EB 减少 active experts，但 linear placement 下 EP load skew 略变差：EB `3.118` vs no-EB top4 `2.970`，说明 active-expert reduction 和 load balancing 是两个目标。
  - 简单 request centroid grouping 不够好；scheduler 需要 set-similarity / load-aware 指标。

#### 新增关键文件

- `codex_coding/src/collect_eb_heteval512_laws.py`
- `codex_coding/results/eb_heteval512_laws_20260428.log`
- `codex_coding/results/eb_heteval512_laws_20260428_rank0.json.gz`
- `codex_coding/results/eb_heteval512_laws_20260428_rank4.json.gz`
- `codex_coding/results/eb_heteval512_laws_20260428_summary.json`
- `codex_coding/results/eb_heteval512_laws_20260428_extended_controls.json`
- `code_building/process_docs/v0.1-init-project/v0.1.15.13-eb_heteval512_law_probe.md`

#### 本轮命令（追加）

- `python3 -m py_compile codex_coding/src/collect_eb_heteval512_laws.py`
- `torchrun --standalone --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 32 --gen-length 32 --output-prefix eb_heteval512_laws_smoke_20260428`
- `torchrun --standalone --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 512 --gen-length 256 --output-prefix eb_heteval512_laws_20260428`

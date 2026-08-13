# Insight Ledger

## Purpose

This is the canonical living ledger for dLLM / MoE / EB optimization insights in this project. It is meant to guide direction selection before implementation.

Use this file to answer:

- Which mechanisms are strongly supported by experiments?
- Which mechanisms were weakened or rejected?
- Which optimization directions follow from the evidence?
- What evidence should be checked before investing in native or Triton implementation?

Process documents remain the source of detailed history. This ledger is the compact maintained index.

## Method Notes

- Monkey-patch implementations are valid for fast mechanism validation.
- Monkey-patch performance should not be treated as the final achievable performance when Python/control overhead dominates.
- A mechanism should only be promoted to native/Triton/vLLM integration when the observed benefit source is large enough to survive implementation overhead.
- For MoE optimizations, distinguish theoretical FLOP/expert-pair savings from wall-clock savings. Current fused MoE kernels are often constrained by expert weight HBM loading.

## Insight Table

| ID | Insight | Mechanism | Data Source | Status | Optimization Implication |
|---:|---|---|---|---|---|
| I1 | Forward count is the highest-leverage dLLM variable. | dLLM decodes a block through repeated full model forwards; removing one forward saves attention, MoE, LMHead, cache work, and distributed communication together. | KC 2026-04-09 to 2026-04-11; BD+IterSmooth experiments | Strongly supported | Decoder, threshold, block iteration, and confidence policy should stay first-class optimization targets. |
| I2 | Cache is not monotonically beneficial in dLLM. | Batch=1 cache paths suffer from cache management and low H100 utilization; large-batch long-context no-cache paths suffer from `O(batch * seq^2)` attention scaling. | KC 2026-04-10 to 2026-04-11 | Strongly supported | Runtime should select cache/no-cache by batch size and context length instead of using one global policy. |
| I3 | MASK routing concentration is a dLLM-specific MoE phenomenon. | MASK tokens share embeddings and produce similar hidden states, especially in shallow layers, concentrating expert choices and causing load imbalance. | KC 2026-04-12; P11 E1-E5 validation | Strongly supported | Use load-aware expert placement, block-start load analysis, and EPLB-style balancing. |
| I4 | Cross-iteration MoE redundancy exists, but direct output reuse is unsafe. | Decoded/stable positions show routing/output similarity, but MoE-output approximation errors propagate through attention and layers. | KC 2026-04-12; v0.1.13 ablations | Mechanism supported; direct skipping rejected | Avoid full-output stable cache. If revisiting, use risk guards, low-precision recompute, or limited/local approximations. |
| I5 | `S_mask` has strong within-block temporal stability. | Cold path constructs an expert budget, and hot/skip paths can reuse it over repeated block forwards; `q_major=1.0` improves stability. | P8 S_mask stability; P11 Insight A; P12g path counts | Strongly supported | Reuse EB metadata and schedule decisions around block boundaries. |
| I6 | `S_mask` stability does not imply hidden/output stability. | `S_mask` only constrains the expert candidate set; hidden states, expert weights, and attention-conditioned outputs can still drift. | Stable cache failures; P12g Scheme3 analysis | Strongly supported | Reuse metadata/schedules, not routed MoE outputs without strong guards. |
| I7 | EB value grows with batch size. | Larger heterogeneous batches activate more unique experts; limiting the candidate expert set reduces active expert count more effectively. | P8 HetEval-128 M-sweep | Supported | EB is more promising for large-batch serving than small-batch single-request optimization. |
| I8 | Fused MoE wall time is often dominated by expert weight HBM loading. | Weight-zero and token-expert-pair pruning do not help unless they reduce unique active experts and weight loads. | P8n NCU; KC 2026-04-14 kernel micro-benchmark | Strongly supported | Optimize active expert count / expert weight loading, not just FLOPs or per-token weights. |
| I9 | Padding-free MoE is not a promising primary direction here. | Padding compute is not the bottleneck; vLLM fused MoE kernels are already H100-tuned and memory-bound. | KC 2026-04-13; P8n | Rejected as primary path | Do not spend more effort on padding-free kernels unless workload changes. |
| I10 | `topk=4` is a real but quality-sensitive signal. | Reducing selected experts from 8 to 4 preserved forward count in key runs and reduced ms/fwd at batch=128, but quality showed mild degradation in at least one verifiable prompt. | P8n routing topk compression | Promising, needs broader quality validation | Worth validating as a native MoE/topk path with larger quality suite. |
| I11 | Top-p expert pruning theoretical savings do not directly imply wall-clock savings. | Top-p reduces token-expert FLOPs, but unique active expert count can remain almost unchanged, so weight loading remains. | KC 2026-04-14; kernel micro-benchmark | Revised downward | Top-p needs active-expert compaction or native kernel support to become wall-clock useful. |
| I12 | Python monkey-patch is a good mechanism-validation tool, not a final performance form. | It enables fast A/B testing, but Python and per-layer control overhead often eats small gains. | v0.1.14 hook overhead; P12g Scheme3 | Methodological insight | Use monkey-patch for feasibility; only native/Triton-integrate mechanisms with large enough benefit sources. |
| I13 | Scheme3 routing-logits communication saving is real but too small as a standalone target. | Dispatch payload falls from about 827 MB/fwd to about 706 MB/fwd, but wall-clock dispatch improves only about 0.45-0.52 ms/fwd. | P12g 8-GPU timing | Mechanism supported; priority lowered | Do not focus on routing-logits communication alone. Combine only if larger structural changes also exist. |
| I14 | dLLM + DP AllToAll has a collective-alignment problem. | Different DP ranks can converge in different numbers of block iterations, while AllToAll collectives require identical call order. | P9 DP AllToAll adaptation | Strongly supported | Multi-GPU dLLM needs block/iteration-level synchronization and scheduler-aware design. |
| I15 | TP/parallelism structure can dominate local MoE patch gains. | dInfer originally duplicated attention under TP-like multi-GPU execution; C11 attention TP and LMHead TP reduced major redundancy. | P10; P11 C11/C12 | Strongly supported | Sequence Parallel / TP / LMHead TP should remain high-priority system directions. |
| I16 | Expert popularity is data-dependent, not a model constant. | Hot expert/GPU patterns differ by dataset; static replication or fixed placement assumptions fail. | P11 Insight C | Strongly supported | Prefer online/adaptive placement or load-aware routing over fixed model-level expert assumptions. |
| I17 | Block boundary is a natural synchronization and planning point. | Shape, block id, cold EB state, cache reset/update, and dLLM iteration state align at block starts. | P11 Insight E; P12 Scheme3 block clock | Architectural insight | Use block starts for safe scheduling, graph capture boundaries, EB cold planning, and synchronization. |
| I18 | Shared expert is important but cannot replace routed experts. | Shared expert contribution is large, but shared-only approximation is not accurate enough; routed expert path remains essential. | KC 2026-04-12; v0.1.14 | Supported | Treat shared expert as a guard or signal, not a replacement for routed MoE. |
| I19 | BSP-MoE validates TP-local MoE token de-duplication, but collective overhead is now the blocker. | Splitting `N=local_bsz*seq_len` across TP ranks reduces dispatch payload from `826.877` to `206.719 MB/fwd`; no-timing e2e improves only `1.40%`, while component timing shows AgRs combine and TP all-gather overhead. | P12j BSP-MoE validation | Mechanism supported; native implementation needed | Keep BSP as a high-value system direction, but next work must optimize collective layout/native integration rather than add Python monkey-patches. |

## Status Legend

| Status | Meaning |
|---|---|
| Strongly supported | Repeatedly validated by experiments or source-level constraints. |
| Supported | Evidence is positive, but scope or quality validation should still be expanded. |
| Promising, needs broader quality validation | Performance signal exists; correctness/quality risk remains. |
| Mechanism supported; direct skipping rejected | The observed phenomenon is real, but the naive optimization path failed. |
| Revised downward | The mechanism remains true but estimated wall-clock value was reduced by later evidence. |
| Rejected as primary path | Do not prioritize unless workload or implementation assumptions change. |
| Methodological insight | Guidance about how to run experiments, not a model/runtime mechanism itself. |
| Architectural insight | Stable structural property that guides system design. |

## Source Index

| Short Name | Source |
|---|---|
| KC | `/home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md` |
| P8 | `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.8-eb_hot_path_optimization_and_batch_scaling.md` |
| P8n | `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.8n-ncu_profiling_tiling_and_multi_gpu.md` |
| P9 | `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.9-dp_alltoall_ep_adaptation.md` |
| P10 | `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.10-deepep_integration_and_attn_tp.md` |
| P11 | `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.11-c11_c12_tp_attention_and_insight_optimization.md` |
| P12g | `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12g-scheme3_8g_timing_and_quality.md` |
| P12j | `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12j-bsp_moe_validation.md` |

## Candidate Direction Buckets

| Direction | Primary Insights | Current Judgment |
|---|---|---|
| Reduce forward count | I1, I2 | Highest leverage; should remain a primary line. |
| Sequence/TP/parallelism structure | I14, I15, I17, I19 | High-value system direction after C11/C12; BSP validates the mechanism but shifts the bottleneck to native collective layout. |
| Native active-expert reduction | I5, I7, I8, I10, I11 | Promising only if it reduces unique active experts or weight loading. |
| Block-stage-aware serving scheduler | I2, I3, I14, I16, I17 | Promising for production serving; needs multi-request validation. |
| Scheme3-style routing-logits communication | I5, I13 | Mechanism is valid but standalone priority is low. |

## Update Rules

When adding or changing an insight:

1. Add the source document or result file to the Source Index if needed.
2. Update the Status field rather than deleting old conclusions.
3. If an optimization path is rejected, preserve the mechanism if it remains true.
4. Append a short update to `/home/wuhang/wuhang/dllm_wh/code_building/progress_diff_summary.md`.
5. If the change affects top-level direction, add a short note to `/home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md`.

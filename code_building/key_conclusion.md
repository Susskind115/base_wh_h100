# Key Conclusions

## Living Insight Ledger

- 方向选择前优先查看 `/home/wuhang/wuhang/dllm_wh/docx/context_index/04_insight_ledger.md`。
- 该文件是当前 dLLM / MoE / EB insight 的 canonical 活文档，维护每条 insight 的机制、数据来源、状态和优化含义。
- 新实验如果改变了 insight 的状态，应优先更新该 ledger，再在本文件追加最高层结论。

## 2026-04-28 (v0.1.15.12p BSP-G2 vLLM SP-Parity Bundle)

- **BSP-G2 已完成实验闭环，但不是新的性能台阶**：
  - C12 no-quality: A `75.428 ms/fwd`，E `71.816`，G `69.676`，G2 `69.661`，F `72.380`
  - G2 vs A 为 `-7.646%`
  - G2 vs G 只快 `0.015 ms/fwd`，属于测量噪声内持平
  - 所以性能参考仍应以 G 为主，而不是把 G2 视作比 G 更快的新路线
- **G2 的有效贡献是源码组织边界更干净**：
  - G2 让 attention path 同时负责 attention-input gather 和 attention-output reduce-scatter
  - component timing 中，G 的 `moe.tp_all_gather=2.662 ms/fwd,count=5054`
  - G2 变为 `attn.input_all_gather=2.573 ms/fwd,count=4788`，残留 `moe.tp_all_gather=0.141 ms/fwd,count=266`
  - 这证明 vLLM/SP-parity 的 boundary ownership 搬运成功
- **G2 没有减少通信字节或同步点**：
  - G 的 `tp_gather_payload=165.375 MB/fwd`
  - G2 变成 `attn_input_gather_payload=156.671 MB/fwd` 加 `tp_gather_payload=8.704 MB/fwd`
  - G/G2 都有 `attn_rs_payload=661.502 MB/fwd` 和 `dispatch_payload=206.719 MB/fwd`
  - 所以 G2 是 bucket 迁移/组织等价，不是 BSP-H/F2 那类 collective 融合
- **component timing 下 G2 反而慢于 G**：
  - G `78.149 ms/fwd`
  - G2 `79.786 ms/fwd`
  - 差值 `+1.637 ms/fwd`
  - 这更支持“G2 是组织方案，不是性能方案”的判断
- **EB/s_mask 与 G2 兼容**：
  - C12 e2e 和 component timing 中 A/E/G/G2/F 全部保持 `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`
  - 小 batch smoke 中 E 仍出现 `hot_skip=893`，但 C12 正式 invariant 正常；记录为小 batch artifact，不影响 C12 结论
- **下一步判断**：
  - 若目标是“现在能吃的性能”，继续以 G 为 measured-best
  - 若目标是“源码下沉时边界更清晰”，G2 的 ownership 设计可作为参考
  - 不应因为 G2 结果直接进入 BSP-H/F2；真正下一步必须减少/fuse 同步点或继续延长 SP lifetime，而不是单纯迁移 gather 归属

## 2026-04-28 (v0.1.15.12q vLLM SP-Parity Inventory)

- **新增持续确认清单**：
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12q-vllm_sp_parity_inventory.md`
- **核心原则**：
  - 不能因为完成 G/G2 就声称 vLLM SP-parity 全部搬完。
  - 必须逐项确认 vLLM 中已有的 SP-MoE、communication backend、compilation SP、residual scattered、CUDA graph/static-size 支持。
- **优先确认项**：
  - LLaDA2 shared expert / dense MLP 在 SP 下是否还有 TP 冗余。
  - 当前环境可用的 vLLM MoE communication backend 及其 SP 支持。
  - vLLM compilation-level sequence parallelism pass 是否能迁移或手工等价。
  - source landing 时 residual scattered metadata、static sizes、CUDA graph 约束。
- **当前判断**：
  - G 仍是 measured-best 性能路径。
  - G2 是 ownership parity，不是性能点。
  - 下一轮讨论/分析应围绕 inventory 表逐项确认，而不是直接开始新代码建设。

## 2026-04-28 (v0.1.15.12r BSP-G3 to G7 vLLM SP Completion)

- **G3 / VSP-06 结论**：
  - vLLM DeepSeek/Llama4 的 shared expert SP 处理核心是 `disable_tp=is_sequence_parallel`，避免 TP Linear 的结果 all-reduce。
  - 当前 LLaDA2-mini 的 shared expert 和 dense-only MLP 都是 replicated `nn.Linear`，不是 TP Linear。
  - BSP-G 已经在 `hs_sp` 上运行 shared expert，因此没有遗漏一个可直接搬的 shared expert 性能点。
- **G4 source-downshift 结论**：
  - 首版源码下沉目标应是 measured-best BSP-G，而不是 G2。
  - G 的核心收益是 attention `RowParallelLinear` 输出从 TP all-reduce/full layout 改为 token-axis reduce-scatter/SP layout。
  - G2 的 attention-input gather ownership 更干净，但没有新性能收益，只作为 source organization 参考。
- **G5 AsyncTP 结论**：
  - vLLM `SequenceParallelismPass` / `AsyncTPPass`、`torch.ops.vllm.*`、`torch.ops.symm_mem.fused_*` 在当前环境可用。
  - 但当前 BSP-G monkey-patch benchmark 没进入 vLLM compile pass manager，所以没有实际吃到 AsyncTP fusion。
  - BSP-G/G2 源码形态概念上接近 `GEMM+ReduceScatter` / `AllGather+GEMM`，source landing 时应保留这种图形态。
- **G6 backend 结论**：
  - AgRs `allgather_reducescatter` 是当前默认和已有性能参考。
  - FlashInfer all2all capability 存在，是唯一较低风险的后续 isolated smoke backend。
  - DeepEP 当前 ABI-broken：`deep_ep._C` 缺 `ncclTeamWorld`，不能测试。
  - PPLX 未安装；FlashInfer Cutlass fused MoE 不是当前 bf16 BSP-G 的直接通信 drop-in。
- **G7 source landing 约束**：
  - source landing 必须显式携带 SP layout metadata 和原始 `N`，不能只靠 tensor shape 推断。
  - static/cudagraph sizes 必须是 TP multiple；C12 `N=8192,tp=4,N_sp=2048` 是安全形状。
  - 避免 forward-time module/buffer mutation，不能延续 monkey-patch 式 per-forward 状态修改。
  - 所有后续 source/backend 实验仍必须保持 C12 path counts：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。

# Key Files Index

## 2026-04-09

- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_vllm_async_tp_patterns.py`
  - 作用：只读 probe，用于复查 vLLM sequence-parallel / AsyncTP 的 op/pass 环境前提，并扫描当前 BSP-G/G2 源码是否具备概念上的 `GEMM+ReduceScatter` / `AllGather+GEMM` pattern 形态。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/vllm_async_tp_pattern_probe_20260428.json`
  - 作用：AsyncTP pattern probe 结果；确认 vLLM pass/op 前提存在，`symm_mem.fused_*` 可注册，但当前 benchmark 未使用 vLLM compile pass manager。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_vllm_moe_backends.py`
  - 作用：只读 backend inventory probe，检查 AgRs/Naive/DeepEP/PPLX/FlashInfer all2all/FlashInfer Cutlass 的模块、env 和 vLLM helper 可用性。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/vllm_moe_backend_inventory_20260428.json`
  - 作用：MoE backend inventory 结果；AgRs 默认可用，FlashInfer all2all capability 可用，DeepEP ABI-broken，PPLX absent，FlashInfer Cutlass 非当前首选。

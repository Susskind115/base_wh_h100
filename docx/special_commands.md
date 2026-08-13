# 专用命令备忘（可持续追加）

更新时间：2026-06-18

## 1. NCU 固定命令（当前已验证）

```bash
sudo /usr/local/cuda-13.0/bin/ncu
```

说明：当前环境下非 sudo 执行会触发 `ERR_NVGPUCTRPERM`，因此后续统一使用上面的完整命令前缀。

## 2. 快速验证（推荐先跑）

```bash
sudo /usr/local/cuda-13.0/bin/ncu --launch-count 1 /home/wuhang/wuhang/rolling_wh/tmp/bin/ncu_smoke_vector_add
```

用途：快速确认 ncu 能启动并成功采样，避免 `--set full` 长时间等待。

## 3. 全量采样（耗时较长）

```bash
sudo /usr/local/cuda-13.0/bin/ncu --set full <cuda_binary> [args...]
```

示例：

```bash
sudo /usr/local/cuda-13.0/bin/ncu --set full /home/wuhang/wuhang/rolling_wh/tmp/bin/ncu_smoke_vector_add
```

## 4. 使用约定

1. 所有 ncu 命令默认写成“`sudo + 绝对路径 ncu`”形式。
2. 先跑快速验证，再决定是否跑 `--set full`。
3. 复盘时若看到未加 sudo 的 ncu 命令，视为不符合当前环境约定。

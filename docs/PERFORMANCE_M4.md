# Kage Performance Tuning on Apple M4

This guide captures the current performance behavior seen on M4 with:

- `MLX_MODEL=mlx-community/Qwen3.5-9B-MLX-4bit`
- `LLM_BACKEND=mlx_vlm`

## What the numbers mean

Recent benchmark output shows two different speed regimes:

- Decode speed is high (`~21 tok/s`).
- End-to-end latency is dominated by prefill/TTFT (`~3-4s`).

Example from `python main.py --bench`:

- `decode @ 21.1-22.2 tok/s`
- `prefill: ~600-800 tok @ ~191-193 tok/s`

Interpretation:

- The model is decoding quickly once generation starts.
- The largest delay is initial prompt processing (system prompt + chat template + user turn).

This is why old aggregate tok/s looked low even though decode throughput was good.

## Backend and model compatibility

`mlx-community/Qwen3.5-9B-MLX-4bit` includes `vision_tower` parameters, so it is a VLM checkpoint.

- Use: `LLM_BACKEND=mlx_vlm`
- Do not use: `LLM_BACKEND=mlx` with this checkpoint

If you only need text and want lower latency, use a text-only checkpoint with `LLM_BACKEND=mlx`.

## Fastest practical settings for text-only usage

For lowest latency in Kage request handling (these are now the default app settings):

- `AGENT_ENABLED=false` (skips routing/tool loop checks)
- `SECOND_BRAIN_ENABLED=false` (no entity memory injection/extraction path)
- `RECENT_TURNS=0` (no chat-history replay)
- `TEMPERATURE=0.0` (deterministic, slightly lower overhead)
- Keep `MLX_MAX_TOKENS` modest (for example `96-160`) if replies can be short

You can still keep your preferred model while applying these.

## Hardware checks

Run these in the same shell/session used by Kage:

```bash
python -c "import platform; print(platform.machine())" && pmset -g | grep -i lowpowermode
```

Expected:

- `arm64`
- low power mode not enabled

Monitor power domains during benchmark:

```bash
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -n 8
```

## Why 16 GB RAM is not the bottleneck here

Observed active MLX memory:

- Qwen3.5-9B-MLX-4bit on `mlx_vlm`: about `5.95 GB`
- Qwen2.5-7B-Instruct-4bit on `mlx`: about `4.29 GB`

Both are well below 16 GB, so current slowdown is not from memory exhaustion.

## Recommended operating profiles

1. Quality-first (current model):
   - `LLM_BACKEND=mlx_vlm`
   - `MLX_MODEL=mlx-community/Qwen3.5-9B-MLX-4bit`
2. Text-speed-first:
   - `LLM_BACKEND=mlx`
   - `MLX_MODEL=<text-only MLX checkpoint>`
   - Optional: set `MLX_DRAFT_MODEL` for speculative decoding (mlx backend only)

## Benchmark command

```bash
python main.py --bench
```

The benchmark now reports:

- TTFT
- total time
- decode tok/s
- prefill token count + prefill tok/s (when backend exposes it)

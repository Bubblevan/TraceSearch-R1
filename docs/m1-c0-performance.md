# M1-C0 runtime evidence

This is a hardware-specific engineering artifact from the current WSL2
single-GPU smoke on an RTX 4090 Laptop GPU with 16 GiB visible memory. It is a
small C0 smoke, not a benchmark and not an algorithmic result.

| profile | cpu_offload_gb | eager | step (s) | rollout throughput | generated tokens |
| --- | ---: | --- | ---: | ---: | ---: |
| baseline | 4.0 | true | ~8.04 | ~30.60 | 246 |
| validated | 0.0 | false | ~3.51 | ~71.72 | 252 |

The comparison is retained as runtime engineering evidence only. The token
counts are approximately comparable, but the observed difference is not
attributed to FlashAttention: vLLM uses its own attention backend here, and no
profiling evidence establishes FlashAttention as the cause.

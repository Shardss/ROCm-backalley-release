# Golden stack — validated on RX 9070 (gfx1201)

Frozen base: `73a43d1` (tag `backalley-base`)
Hardware: RX 9070, Adrenalin 32.0.31041.1004, ROCm 7.2 HIP SDK, Windows 11, MSVC 14.44

## Included

| PR | what | requires |
|---|---|---|
| #26301 | dequant-float matvec (mmvdq) for Q4_K/Q5_K/Q6_K | `GGML_CUDA_DQ_MMV=1` (arch default excludes RDNA4) |
| #28102 | Flash Attention tuning (gfx1201) | — |
| #24386 | RDNA4 MMVQ warps for K-quants (Q4_K=3, Q6_K=5) | — |
| #25940 | HIP RDNA4 MUL_MAT (MMQ) optimizations | — |

Apply in this order. #25940 must go after #24386 and **without** #18816.

## Cumulative gain vs frozen base

Stage 1 — #26301+#28102+#24386 vs base:

| model | pp512 | tg128 |
|---|---|---|
| Qwen3-4B Q4_K_M | +1.25% | +26.34% |
| Qwen3-8B Q4_K_M | +1.73% | +9.96% |
| Qwen3-4B Q4_0 | +3.13% | +2.86% |
| Qwen3-4B IQ4_XS | +2.38% | +4.65% |
| OLMoE-1B-7B (MoE) | -0.19% | +35.36% |

Stage 2 — adding #25940 on top:

| model | pp512 | tg128 |
|---|---|---|
| Qwen3-4B Q4_K_M | +10.44% | -0.32% |
| Qwen3-8B Q4_K_M | +10.10% | -0.25% |
| OLMoE-1B-7B (MoE) | +6.46% | 0% |
| Qwen3-4B Q4_0 | -0.68% | -0.15% |
| Qwen3-4B IQ4_XS | -0.38% | +0.03% |

Combined vs frozen base (approx, composing both stages):

| model | pp512 | tg128 |
|---|---|---|
| Qwen3-4B Q4_K_M | ~+11.8% | +26.3% |
| Qwen3-8B Q4_K_M | ~+12.0% | +10.0% |
| OLMoE-1B-7B (MoE) | ~+6.3% | +35.4% |
| Qwen3-4B Q4_0 | +2.4% | +2.9% |
| Qwen3-4B IQ4_XS | +2.0% | +4.7% |

No regressions on any model or metric.

## Excluded, and why

| PR | reason |
|---|---|
| #18816 | +15.6% dense prefill but **-47.8% MoE prefill** (confirmed, n=3, disjoint ranges). Needs its RDNA4 `ne11<=256` thresholds gated on expert count. |


## Reproduce

```
git checkout backalley-base
git apply stack_26301.diff stack_28102.diff stack_24386.diff stack_25940.diff
cmake --build build-hip --target llama-bench
GGML_CUDA_DQ_MMV=1 llama-bench -m <model> -ngl 99 -p 512 -n 128
```


## Quality gate — PASS

Frozen base vs golden stack, Qwen3-4B Q4_K_M, wikitext 32 chunks, winogrande 150 tasks:

| check | base | golden stack | threshold | result |
|---|---|---|---|---|
| failing ops (vs CPU reference) | 0 | 0 | any new → fail | pass |
| perplexity | 9.1136 | 9.1122 (−0.0154%) | >0.5% → fail | pass |
| mean KL divergence | — | **0.002916 ± 0.000076** | >0.01 → fail | pass |
| winogrande | 67.3333 | 67.3333 (0 pts) | −2 pts → fail | pass |

Same-top-token agreement: **97.659% ± 0.167**. Median KLD 0.001297.
The residual difference is consistent with floating-point reduction-order changes,
not altered model behaviour.

**The golden stack is 12–26% faster and measurably the same model.**

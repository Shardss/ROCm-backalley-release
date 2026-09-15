# ROCm backalley — Release v4

**v4 is a long-context release.** Flash attention on this card was spilling registers inside
the loop that walks the KV cache, so the cost grew with every token of context. Two head
dimensions were affected, and both are fixed here.

Nothing in v4 helps an empty cache much. Everything in it helps a long one.

| model | prefill at 8k | prefill at 16k |
|---|---|---|
| Qwen3-4B Q4_K_M | **+14.2%** | **+22.5%** |
| Qwen3-8B Q4_K_M | **+9.0%** | — |
| Qwen3.8-27B IQ4_XS | **+13.3%** | **+20.3%** |

Decode is unchanged except where a separate change lands: **gemma-26B IQ4_XS decode +6.0%**.

---

## What v4 adds over v3

| patch | what it does |
|---|---|
| `v4_fa_spill.diff` | head_dim 128 flash attention: makes the mask a runtime branch on AMD, and halves `nbatch_fa`. Cuts register spills 210 → 37. |
| `v4_packed_q8_1_layout.diff` | extends #23685's packed Q8_1 activation layout to IQ4_XS and Q3_K, which it did not cover. |
| `pr26419.diff` | upstream PR: enables MMA flash attention for head_dim 256 on RDNA4. Abandoned at v3 as a semantic conflict; ported here. |
| `v4_hd256_retune.diff` | retunes the head_dim-256 kernel config for gfx1201. Cuts its spills 1,129 → 149. |

---

## The finding: register spills in flash attention

Each GPU thread has a fixed budget of registers. A kernel that needs more live values than it
has spills the excess to memory and reloads it. In these kernels that traffic sits **inside the
loop over the KV cache**, so its cost scales with context depth — which is why it hid: the
upstream op benchmarks never run head_dim 128 at long KV.

The compiler reports it directly (`-Rpass-analysis=kernel-resource-usage`):

| kernel | v3 | v4 |
|---|---|---|
| head_dim 128 | 210 spills | **37** |
| head_dim 256 | *no code — kernel disabled on AMD* | **149** |
| every other head dim | 0 | 0 |

head_dim 128 covers most models. head_dim 256 is Qwen3.8-27B.

### PR 26419 alone is worth nothing

26419 switches the head_dim-256 MMA kernel on. Measured on top of v4, by itself, on the very
model it targets:

| depth | with 26419, stock config |
|---|---|
| 0 | −0.74% |
| 4096 | −0.46% |
| 8192 | +0.35% |

It enables a kernel that arrives spilling 1,129 registers, so it starts crippled. The config
retune is what converts it:

| config | spills | prefill at 8k | prefill at 16k |
|---|---|---|---|
| 26419 as shipped | 1,129 | +0.4% | — |
| `nbatch_fa` 32 only | 356 | +2.7% | +0.1% |
| `nbatch_fa` 32, K/V batch 64, **Q in LDS** | **149** | **+14.0%** | **+20.4%** |

Moving the query tile out of registers into shared memory is the decisive step, and it is
*counterproductive on its own* (1,129 → 1,247 spills) — it only pays once `nbatch_fa` is halved.
The obvious fix, copying what cured head_dim 128, would have collected 2.7% and stopped.

---

## Results — packaged v3 vs packaged v4

Run from the release archives themselves, `llama-bench`, 3 repetitions, `-fa 1`, `-p 512 -n 128`.
`-d` is how many tokens are already in the KV cache.

### Prefill

| model | depth | KV | v3 | v4 | change |
|---|---|---|---|---|---|
| Qwen3.8-27B IQ4_XS | 0 | f16 | 1,068.93 | 1,074.48 | +0.52% |
| Qwen3.8-27B IQ4_XS | 4,096 | f16 | 920.55 | 989.47 | **+7.49%** |
| Qwen3.8-27B IQ4_XS | 8,192 | q8_0 | 802.90 | 909.97 | **+13.33%** |
| Qwen3.8-27B IQ4_XS | 16,384 | q8_0 | 642.50 | 773.06 | **+20.32%** |
| Qwen3-4B Q4_K_M | 0 | f16 | 6,220.63 | 6,361.86 | +2.27% |
| Qwen3-4B Q4_K_M | 8,192 | f16 | 3,043.25 | 3,476.33 | **+14.23%** |
| Qwen3-4B Q4_K_M | 16,384 | f16 | 1,887.28 | 2,312.19 | **+22.51%** |
| Qwen3-8B Q4_K_M | 8,192 | f16 | 2,297.57 | 2,504.06 | **+8.99%** |
| gemma-26B IQ4_XS | 0 | f16 | 4,053.28 | 4,267.22 | +5.28% |
| Qwen3-Coder-30B Q3_K_XL | 0 | f16 | 3,277.24 | 3,313.63 | +1.11% |

### Decode

| model | v3 | v4 | change |
|---|---|---|---|
| gemma-26B IQ4_XS | — | — | **+6.00%** |
| Qwen3.8-27B IQ4_XS | — | — | +0.78% to +0.97% |
| Qwen3-Coder-30B Q3_K_XL | — | — | +0.37% |
| Qwen3-4B / Qwen3-8B Q4_K_M | — | — | −0.18% to +0.15% |

The gemma-26B figure is the IQ4_XS packed-layout patch. #23685 brought the packed Q8_1
activation path to K-quants in v2 but never covered the IQ types; v4 closes that.

### What did *not* improve

- **Empty-cache prefill** barely moves (+0.5% to +2.3%). The bug was depth-dependent; so is the fix.
- **Decode on K-quant models** is unchanged. Decode is bandwidth-bound and these are attention fixes.
- **head_dim-128 models gain nothing from PR 26419.** Measured separately at +0.5 to +0.8%
  against a 0.34% run-to-run noise floor. Their gain comes from `v4_fa_spill.diff`.

---

## Quality gate

Same gate v3 passed — `gate3.ps1`, fails closed, against the frozen base.

```
verdict     PASS
ops         15738 / 15738, 0 failing
perplexity  9.1241
mean KLD    0.002924   top-1 agreement 97.806%
winogrande  66.6667%
batched     npl 1,2,3,4,5,6,8 - no cell below -5% vs base
```

**v4 is not numerically identical to the base, and v3 was.** This is a real difference and is
stated here rather than buried:

| | perplexity | mean KLD | top-1 agreement | winogrande |
|---|---|---|---|---|
| base | 9.1136 | 0 | 100% | 67.33% |
| v3 | 9.1136 | 0 | 100% | 67.33% |
| **v4** | **9.1241** | **0.002924** | **97.806%** | 66.67% |

Perplexity moves +0.115%. Winogrande differs by a single task out of 150. About 2.2% of tokens
pick a different top token than the base does. The cause is the flash-attention changes altering
accumulation order. We judged that acceptable for a 13–22% long-context gain; if you need
bit-identical output to upstream, stay on v3.

Perplexity on the 27B, separately: v3 5.6061 → v4 5.6003 (−0.10%).

---

## Limitations

- **One card, one OS, one driver.** RX 9070 (gfx1201), Windows 11, Adrenalin 32.0.31041.1004,
  ROCm 7.2. Untested on 9070 XT, 9060, R9700, or Linux.
- **The head_dim-256 work is validated on one model.** Qwen3.8-27B is the only head_dim-256
  model here. The config line lives in the shared RDNA table, but only RDNA4 can reach that
  path (26419 gates it on `GGML_CUDA_CC_IS_RDNA4`), so RDNA3 is untouched by construction —
  untested rather than unaffected.
- **The prebuilt archive is gfx1201-only.** Tensile libraries are filtered to gfx1201.
- **Depth beyond 16k is not measured on the 27B**, because it does not fit — see below.

### Context length and VRAM on a 16 GB card

Qwen3.8-27B stores **260 KB of KV cache per token** (65 layers × 4 KV heads × 256 head dim,
K and V, f16). That is 4.2 GB at 16k and 13.3 GB at 50k — as much as the model itself.

| context | f16 KV | + 13.3 GB model | fits in 16 GB? |
|---|---|---|---|
| 7k | 1.9 GB | 15.2 GB | just |
| 16k | 4.3 GB | 17.6 GB | no — use `-ctk q8_0 -ctv q8_0` |
| 50k | 13.3 GB | 26.6 GB | no |

Past roughly 1 GB of free VRAM the driver spills the cache to system RAM and decode drops by
more than 10x. Halve the cache with `-ctk q8_0 -ctv q8_0` before you get there. The 8k and 16k
figures above use q8_0 KV for this reason.

---

## Building

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 73a43d1

# the v3 stack, in this order
git apply ../patches/stack_26301.diff
git apply ../patches/stack_24386.diff
git apply ../patches/stack_25940.diff
git apply ../patches/rel_25206.diff
git apply ../patches/rel_27248.diff
git apply ../patches/rel_26504.diff
git apply ../patches/rel_28477.diff
git apply ../patches/rel_28079.diff
git apply ../patches/rel_27269_new.diff
git apply ../patches/pr28552.diff
git apply ../patches/pr23685_ported.diff
git apply ../patches/cand_RDNA4_MMVQ_XOVER.diff
git apply ../patches/pr28398.diff
git apply ../patches/fix_Q6K_always_mmq.diff

# v4, in this order
git apply ../patches/v4_fa_spill.diff
git apply ../patches/v4_packed_q8_1_layout.diff
git apply ../patches/pr26419.diff
git apply ../patches/v4_hd256_retune.diff
```

`v4_hd256_retune.diff` must come after `pr26419.diff` — it retunes config lines that only
matter once 26419 has enabled that kernel.

Build with the ROCm 7.2 HIP SDK and **VS2022**. MSVC 14.51 (VS2026) fails on ROCm 7.2 with
`__clang_cuda_math_forward_declares.h` errors — see upstream #24929.

```
cmake -S . -B build-hip -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 ^
  -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang.exe" ^
  -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang++.exe" ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j 20
```

**Check that your build actually changed.** Ninja compares timestamps, so a source file
restored or copied in with an older mtime than its object file is silently skipped — we lost
a full measurement round to exactly that. Confirm `ggml-hip.dll` differs from your baseline
by content hash before trusting any number.

### Required environment

```
set GGML_CUDA_DQ_MMV=1
set GGML_CUDA_DQ_Q6K=1
set HIP_VISIBLE_DEVICES=0
set ROCR_VISIBLE_DEVICES=0
```

Unchanged from v3. Without the first two you lose roughly a third of the decode gain.

### Verifying

`test-backend-ops.exe` → expect `Backend ROCm0: OK`. Run it five times; a single
`ADD(type=f16)` failure near 1.0e-07 is a known upstream tolerance issue and occurs on
unpatched llama.cpp at a similar rate.

---

## Changes that are ours, not upstream

- `v4_fa_spill.diff` — written here. The mask rewrite is AMD-guarded: on NVIDIA the same
  change makes register spilling roughly 50x worse, which we measured before guarding it.
- `v4_packed_q8_1_layout.diff` — written here, extending #23685's design to IQ4_XS and Q3_K.
- `v4_hd256_retune.diff` — written here, on top of upstream #26419.
- `fix_Q6K_always_mmq.diff` — carried from v3.
- `cand_RDNA4_MMVQ_XOVER.diff` — carried from v2.

Everything else is an upstream contributor's work. Links go to the original PRs; please credit
their authors.

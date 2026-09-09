# ROCm backalley — Release v2

**v2 removes one patch from v1 and adds nine.**

The removal is the important part. **#28102 breaks flash attention** on split key/value head
dimensions, and v1's quality gate did not catch it because that gate ran once and never exercised the
decode path. If you run an MLA-architecture model on v1, move to v2.

**v1 is deliberately left as published.** It has already been distributed, and rewriting a release
people may be running is worse than shipping a clear correction. v2 is that correction.

---

## What changed

### Removed

| patch | why |
|---|---|
| [#28102](https://github.com/ggml-org/llama.cpp/pull/28102) CUDA/HIP Flash Attention tuning (gfx1201) | **Regression.** Breaks four `FLASH_ATTN_EXT` shapes at `hsk=192, hsv=128, kv_view=1`. Errors 0.000815 / 0.001260 / 0.002247 / 0.003965 against a 0.000500 tolerance — 1.6× to 8× over. The frozen base runs all four shapes and reports OK, so this is a regression, not new coverage. |

Isolated by removing each v1 patch in turn and running the suite 12 times per configuration. Dropping
#26301, #24386 or #25940 leaves the failures; dropping #28102 removes them entirely.

Removing it costs roughly 4% of prefill on most models and 7.3% on gemma-26B. Decode and quantized-KV
gains are unaffected. That price is worth paying: none of our benchmark models use that attention
shape, which is precisely why the fault went unnoticed, and anyone running a DeepSeek-style model would
have hit it silently.

### Added

| patch | what it does | measured on gfx1201 |
|---|---|---|
| [#25206](https://github.com/ggml-org/llama.cpp/pull/25206) | optimize RWKV7 inference by fusing graph operators | rwkv7-1.5B pp512 **+9.5%**, tg128 **+7.7%** |
| [#27248](https://github.com/ggml-org/llama.cpp/pull/27248) | CUDA support for `q4_1`, `iq4_nl`, `q5_0`, `q5_1` KV-cache types | quantized-KV decode **+147…168%**; f16/q8_0/q4_0 controls flat |
| [#27269](https://github.com/ggml-org/llama.cpp/pull/27269) | `q8_0`-K / `q4_0`-V flash attention vector kernels. **Needs a one-line HIP fix** — see below | q8_0/q4_0 decode 42.35 → 117.11 t/s (**+176.5%**) |
| [#23685](https://github.com/ggml-org/llama.cpp/pull/23685) | 4× packed Q8_1 activation for Q4_K_M in MMVQ (+Q5_K/Q6_K). **Hand-ported** — see below | tg128 **+31.9%** Qwen3-4B, **+17.3%** Qwen3-8B, **+54.1%** OLMoE |
| [#28552](https://github.com/ggml-org/llama.cpp/pull/28552) | size routed MoE MMQ N-tiles from typical expert width on RDNA3/RDNA4 | pp512 **+14.6%** OLMoE, **+10.6%** gemma-26B; dense control flat |
| [#28398](https://github.com/ggml-org/llama.cpp/pull/28398) | hardware `v_perm_b32` for Q1_0 vec_dot on AMD | tg128 **+77.2%** on Q1_0. Validated in v1 but left out; now included |
| **RDNA4-MMVQ-XOVER** | **our own change**, not an upstream PR — see below | batched decode B8 **+67.6%** Qwen3-4B, **+104.8%** Qwen3-8B |
| [#26504](https://github.com/ggml-org/llama.cpp/pull/26504) | non-contiguous tensors in the CEIL op | no speed change; **+5 passing ops**, 4 moved out of NOT SUPPORTED |
| [#28477](https://github.com/ggml-org/llama.cpp/pull/28477) | strided ABS for F16 and F32 | no speed change; **+13 passing ops**, 4 moved out of NOT SUPPORTED |

The last two buy no speed at all. They are in because this is a release other people run: they let it
execute shapes the base refuses, and capability counts as much as throughput.

### Rejected

| candidate | verdict |
|---|---|
| DPP warp reduction (our own experiment) | Fails `test-backend-ops` deterministically — 3 of 3 runs — breaking `GATED_DELTA_NET`, `MUL_MAT_VEC_FUSION` and `TOPK_MOE`, with passing cases collapsing from ~14697 to ~12480. Measured gain was +0.5–1.2%, at or below the ~0.95% noise floor. It had been cleared twice by a defective correctness checker (see below). |

---

## Three changes that are ours, not upstream

Stated plainly so nobody mistakes them for reviewed upstream work.

**RDNA4-MMVQ-XOVER** — `ggml_cuda_should_use_mmvq()` carries measured per-architecture tables for Ada,
Blackwell, GB10, Orin, CDNA1 and CDNA2, but **none for RDNA**, so RDNA4 fell through to a permissive
`ne11 <= 8`. Measurement shows MMQ beats MMVQ for K-quants once `ne11` reaches 4, so the table now
returns `ne11 <= 3` for `Q4_K` and `Q6_K`. Worth +67.6% / +104.8% at batch 8.

**#23685, hand-ported.** The upstream patch predates template parameters the base has since gained
(`small_k`, `halve_iters`); `mmvq.cu` rejected 7 hunks. The port threads those parameters through the
policy struct. Two deliberate divergences from the PR: it keeps `ggml_cuda_kernel_launch` where the PR
reverts to raw `<<<>>>` (which would strip PDL), and it restores `ggml_cuda_pdl_sync()`.

**#27269 needs a one-line HIP fix.** The PR adds `fattn-vec-instance-q8_0-q4_0.cu` to the **CUDA**
instance list only. The HIP build has its own list in `ggml/src/ggml-hip/CMakeLists.txt`, so without the
matching line the build fails to link. `cand_27269_with_hipfix.diff` carries both.

---

## Results

All figures against the frozen base, `73a43d1`, on an RX 9070 (gfx1201).

| model | pp512 | tg128 |
|---|---|---|
| Qwen3-4B Q4_K_M | +8.1% | **+32.9%** |
| Qwen3-8B Q4_K_M | +7.2% | +16.5% |
| OLMoE-1B-7B (MoE) | **+19.3%** | **+54.4%** |
| gemma-26B-A4B IQ4_XS | +8.0% | +2.6% |
| rwkv7-1.5B | +11.1% | +34.1% |
| Qwen3-4B Q1_0 | **−2.8%** | **+83.2%** |

`Q1_0` prefill is **2.8% slower** than the base. It is the one regression in this release, on a quant
almost nobody ships, and the same model gains +83.2% on decode.

**Quantized KV cache**, Qwen3-4B Q4_K_M, `-fa 1`, tg32:

| K/V | base | v2 | gain |
|---|---|---|---|
| q4_1 / q4_1 | 42.86 | 147.57 | **+244.3%** |
| q5_0 / q5_0 | 42.50 | 145.70 | **+242.8%** |
| q5_1 / q5_1 | 42.70 | 146.10 | **+242.2%** |
| iq4_nl / iq4_nl | 40.18 | 121.00 | **+201.1%** |
| q8_0 / q4_0 | 43.02 | 153.42 | **+256.6%** |

Five of eight quantized-KV configurations were running about 3× slower than they should. #27248 and
#27269 restore them.

Batched decode, 128 prompt / 64 generate: **693.8 t/s at 8 parallel sequences.**

Op coverage: **OK 14726** against the base's 14692, `NOT SUPPORTED` down from 7579 to 7568.

---

## Quality gate

v1's gate exercised only batched work — an `ne11` probe showed **1 of 8282** mul_mat dispatches ran at
decode width, which is how a flash-attention regression passed it. v2 adds a decode leg (`-b 512 -ub 1`)
where the probe reads **7227 of 7227**.

| check | base | v2 | threshold | result |
|---|---|---|---|---|
| perplexity, batched | 9.1136 | 9.1194 (+0.0636%) | >0.5% | **pass** |
| perplexity, decode | 12.4893 | 12.4687 (−0.1649%) | >0.5% | **pass** |
| mean KLD, batched | — | 0.002894 (same-top 97.929%) | >0.01 | **pass** |
| mean KLD, decode | — | 0.00199 (same-top 98.333%) | >0.01 | **pass** |

Every perplexity figure was measured twice and reproduced to four decimal places.

### Correctness, measured rather than assumed

| | full runs failed | families |
|---|---|---|
| **v2** | **3 of 20** | ADD_ADD ×2, ADD ×1 |
| frozen base, unpatched | 4 of 20 | ADD_ADD ×2, ADD ×2, MUL_MAT ×1 |

**v2 fails less often than unpatched upstream llama.cpp.** Every failure on both sides is a
marginal-tolerance case.

Two faults in the old method were found and fixed while producing this release, and both are worth
knowing if you rely on `test-backend-ops` yourself:

1. **The checker was wrong.** It counted only `OP(...): FAIL` lines and never read the exit code or the
   `Backend <name>: OK|FAIL` summary. `test-backend-ops` lists backend-level failures as bare descriptors
   after `Failing tests:`, which that checker ignored — so a broken build could report zero failing ops.
   Corrected implementation: `scripts/opscheck.ps1`.
2. **One run proves nothing.** `init_tensor_uniform()` seeds from `std::random_device`, so every run uses
   different input data. Stock llama.cpp fails 4 of 20 full runs here on an f16 `ADD` case sitting at
   1.00–1.07e-07 against a 1.0e-07 tolerance. A single-run gate rejects unpatched upstream about 20% of
   the time.

### Known, unexplained

One `MUL_MAT_ID(type_a=q4_K,type_b=f32,n_mats=4,n_used=1,b=0,m=512,n=1,k=256)` failure was seen during
gating and **never recurred** — 0 in 28 later full runs, 0 in 40 filtered runs. The frozen base produces
the same class of event unpatched: `MUL_MAT(q5_1, n=1)` at 0.000509 against 0.000500.

**The gate logged the case string but not its error margin, so this is strong supporting evidence, not
proof.** Rate is below 1 in 28; cause unknown. Reported rather than rounded down to "fluke".

---

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout 73a43d1 && git tag backalley-base

# v1 base, minus #28102
git apply ../patches/stack_26301.diff
git apply ../patches/stack_24386.diff
git apply ../patches/stack_25940.diff

# v2 additions, in this order
git apply ../patches/rel_25206.diff
git apply ../patches/rel_27248.diff
git apply ../patches/rel_26504.diff
git apply ../patches/rel_28477.diff
git apply ../patches/cand_27269_with_hipfix.diff
git apply ../patches/pr28552.diff
git apply ../patches/pr23685_ported.diff
git apply ../patches/cand_RDNA4_MMVQ_XOVER.diff
git apply ../patches/pr28398.diff

cmake -S . -B build-hip -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang++.exe" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF
cmake --build build-hip --target llama-bench llama-batched-bench test-backend-ops
```

That should touch **28 tracked files**. If it does not, a patch failed to apply — check, rather than
assume, because `git apply` failures are easy to swallow by accident.

### Required environment

```
set GGML_CUDA_DQ_MMV=1     # without it you lose roughly a third of the decode gain
set GGML_CUDA_DQ_Q6K=1
set HIP_VISIBLE_DEVICES=0
set ROCR_VISIBLE_DEVICES=0
```

`#26301`'s upstream `arch_default` is RDNA3.5 and **excludes RDNA4**, so on a 9070 the path is compiled
but never taken unless `GGML_CUDA_DQ_MMV=1` is set.

### Verifying your build

```
build-hip\bin\test-backend-ops.exe
```

Expect `Backend ROCm0: OK` and roughly **14726** passing cases. **Run it at least five times.** A single
failure on `ADD(type=f16,...)` or `ADD_ADD(type=f16,...)` with an error near 1.0e-07 is the upstream
tolerance issue described above and occurs on unpatched llama.cpp at a similar rate — it is not a
problem with this build. Anything reproducible, or far over tolerance, is.

---

## Same limitations as v1

One card, one OS, one driver: RX 9070 (gfx1201), Windows 11, Adrenalin 32.0.31041.1004, ROCm 7.2.
Untested on 9070 XT, 9060, R9700 or Linux. Six models. Not an exhaustive quality suite.

Every optimisation here was written by llama.cpp contributors, except the three changes named above as
ours. This project contributes measurement, porting and evidence. Links go to the original PRs; please
credit their authors.

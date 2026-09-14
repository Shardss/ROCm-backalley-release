# ROCm backalley — Release v3

**v3 fixes a prefill regression that shipped in both v1 and v2.**

`#25940`'s RDNA4 MMQ-versus-hipBLAS table sends `Q6_K` matmuls to hipBLAS above `ne11 = 256`.
On gfx1201 that path is **2.6x to 6.7x slower than MMQ**, and it has been in every release to
date. Batched prefill on `Qwen3-4B Q4_K_M` lost up to **85%** against the unpatched base.
Neither release's quality gate measured the shape it lived in.

v1 and v2 are **left as published**. They have been distributed and rewriting them is worse
than shipping a clear correction. v3 is that correction.

---

## The regression

Measured on the frozen base `73a43d1`, RX 9070 (gfx1201), `llama-batched-bench`,
`Qwen3-4B Q4_K_M`, `-fa 1 -c 8192 -b 2048 -ub 512 -npp 128 -ntg 64`, prefill throughput:

| ne11 | base | v2 (shipped) | v3 | v2 vs base | v3 vs base |
|---|---|---|---|---|---|
| 128 | 3,690.75 | 3,819.19 | 3,703.13 | +3.5% | +0.3% |
| 256 | 4,980.85 | 5,326.57 | 5,322.14 | +6.9% | +6.9% |
| 384 | 5,376.65 | **793.92** | 5,767.70 | **−85.2%** | +7.3% |
| 512 | 5,944.91 | **2,352.14** | 6,366.96 | **−60.4%** | +7.1% |
| 640 | 5,433.94 | **1,204.93** | 5,785.50 | **−77.8%** | +6.5% |
| 768 | 5,570.59 | **1,444.57** | 5,940.43 | **−74.1%** | +6.6% |
| 1024 | 6,190.17 | **3,555.84** | 6,583.93 | **−42.6%** | +6.4% |

Decode (`S_TG`) is unaffected in every arm, so this is prefill-specific.

**How it was found.** A batch-size sweep, which no previous gate ran. `npl=4` looked anomalous
in a single un-repeated measurement; three repetitions per cell confirmed it (base 6,122 ±2.6%,
v2 2,337 ±1.1%, distributions non-overlapping).

**How it was attributed.** Rebuilding the stack four times, each with one candidate patch
removed:

| dropped | ne11=512 prefill | verdict |
|---|---|---|
| `cand_RDNA4_MMVQ_XOVER` | 2,221 | still regressed |
| `pr23685_ported` | 2,295 | still regressed |
| `stack_25940` | **6,154** | **restored** |
| `pr28552` | 2,295 | still regressed |

**Why it is `Q6_K` and not `Q4_K`.** `Qwen3-4B Q4_K_M` contains both. `#25940` sets
`Q4_K ≤ 512` but `Q6_K ≤ 256`, so `Q6_K` crosses first — which is why the worst cell is
`ne11 = 384`, below `Q4_K`'s own threshold. Raising only the `Q6_K` bound restores every cell.

**The fix**, in `ggml_cuda_should_use_mmq()`:

```c
case GGML_TYPE_Q6_K:
    // gfx1201: the hipBLAS path collapses batched prefill
    // (-85% at ne11=384). Measured: MMQ wins at every ne11.
    return true;
case GGML_TYPE_Q5_1:
    // left at the upstream threshold - not measured here,
    // none of the benchmark models use Q5_1.
    return ne11 <= 256;
```

This is **ours, not upstream**, and worth reporting back to `#25940`: the `Q6_K ≤ 256`
threshold is wrong for gfx1201.

### The trade

Routing `Q6_K` to hipBLAS *helps* single-sequence prefill. Removing it costs some of that:

| Qwen3-4B Q4_K_M | v1 | v2 | v3 |
|---|---|---|---|
| pp512 vs base | +12.8% | +8.5% | +4.8% |

We consider ~4 points of single-prompt prefill a fair price for removing a 43–85% hole across
every batched shape. **If you only ever run one prompt at a time, v2 was faster for you.**
If anything runs parallel sequences — a server, an agent, batched work — v2 was broken.

---

## Also changed

**`#27269` retired into upstream.** v2 carried `cand_27269_with_hipfix.diff`, a hand-written
line adding `fattn-vec-instance-q8_0-q4_0.cu` to the HIP instance list. Upstream `#28079`
replaced the hardcoded lists with a shared generator (`ggml_cuda_fattn_vec_instances()`), and
`#27269` collapsed to a two-line change putting `q8_0-q4_0` in the default set. v3 takes both
and **drops our patch**. One fewer non-upstream divergence.

Verified rather than assumed: `cand_27269_with_hipfix.diff` fails to apply on top of `#28079`
at `fattn.cu:453` and `ggml-hip/CMakeLists.txt:79`; the replacement applies clean.

**Op coverage: 14,692 → 15,738** (+1,046 cases moved out of `NOT SUPPORTED`).

---

## Results

All figures against the frozen base `73a43d1`, RX 9070 (gfx1201), idle machine,
`llama-bench -ngl 99 -p 512 -n 128 -r 5`.

### Prefill (pp512), % vs base

| model | base t/s | v1 | v2 | v3 |
|---|---|---|---|---|
| Qwen3-4B Q4_K_M | 6,014.74 | +12.8% | +8.5% | +4.8% |
| Qwen3-8B Q4_K_M | 3,497.78 | +12.1% | +10.5% | +6.1% |
| OLMoE-1B-7B (MoE) | 10,106.31 | +4.2% | **+19.1%** | **+17.0%** |
| gemma-26B-A4B IQ4_XS (MoE) | 3,754.43 | +4.4% | +10.7% | +10.9% |
| rwkv7-1.5B | 7,908.35 | +0.3% | +8.9% | +10.3% |
| Qwen3-4B Q1_0 | 4,858.56 | +1.9% | −0.0% | −0.2% |
| Qwen3.8-27B IQ4_XS (dense) | 1,091.14 | +1.1% | +0.0% | −0.2% |

### Decode (tg128), % vs base

| model | base t/s | v1 | v2 | v3 |
|---|---|---|---|---|
| Qwen3-4B Q4_K_M | 125.77 | +25.0% | +33.4% | **+34.0%** |
| Qwen3-8B Q4_K_M | 89.38 | +9.4% | +17.3% | +17.2% |
| OLMoE-1B-7B (MoE) | 246.68 | +34.7% | +53.4% | **+54.1%** |
| gemma-26B-A4B IQ4_XS (MoE) | 107.38 | +3.3% | +3.2% | +3.5% |
| rwkv7-1.5B | 175.49 | +20.6% | +33.6% | +32.9% |
| Qwen3-4B Q1_0 | 142.78 | +4.0% | +84.9% | **+86.1%** |
| Qwen3.8-27B IQ4_XS (dense) | 29.63 | +4.1% | +5.5% | +5.4% |

### Quantized KV cache — `Qwen3-4B Q4_K_M`, `-fa 1`, tg32

| K/V | base | v2 | v3 | v3 vs base |
|---|---|---|---|---|
| f16 / f16 | 125.38 | 165.90 | 165.41 | +31.9% |
| q8_0 / q8_0 | 118.33 | 154.97 | 155.75 | +31.6% |
| q4_0 / q4_0 | 117.60 | 153.74 | 154.28 | +31.2% |
| q4_1 / q4_1 | 42.67 | 149.25 | 149.92 | **+251.4%** |
| q5_0 / q5_0 | 42.63 | 147.06 | 149.16 | **+249.9%** |
| q5_1 / q5_1 | 42.82 | 148.63 | 151.50 | **+253.8%** |
| iq4_nl / iq4_nl | 39.38 | 111.03 | 111.37 | +182.8% |
| q8_0 / q4_0 | 42.93 | 154.53 | 154.03 | **+258.8%** |

Five of eight quantized-KV configurations run ~3x slower than they should on the base.
`#27248` and `#27269` restore them; v3 preserves that intact.

### Batched decode — `S_TG`, same configuration

| ne11 | base | v2 | v3 | v3 vs base |
|---|---|---|---|---|
| 128 | 124.02 | 166.45 | 166.14 | +34.0% |
| 512 | 348.92 | 360.33 | 360.51 | +3.3% |
| 768 | 390.96 | 527.22 | 528.38 | +35.1% |
| 1024 | 403.95 | 686.05 | 685.72 | **+69.8%** |

---

## Quality gate

`gate3.ps1` — five checks, fails closed. New in v3: **a batched throughput sweep with
repetitions**, because v1 and v2 both shipped regressions in shapes their gates never measured.

| check | base | v3 | threshold | result |
|---|---|---|---|---|
| op correctness | 14,692 / 14,692 | 15,738 / 15,738 | no new failures | **pass** |
| perplexity (wikitext-2, 32 chunks) | 9.1136 | 9.1136 | >0.5% | **pass** (Δ 0.0000%) |
| mean KL divergence | — | 0.000000 | >0.01 | **pass** |
| same top token | — | 100.000% | — | **pass** |
| winogrande (150 tasks) | 67.3333% | 67.3333% | >2 points | **pass** (Δ 0.000) |
| batched sweep `npl` 1–8 | reference | +6.4% … +7.8% prefill | any cell < −5% | **pass** |

`test-backend-ops` was run five times on the shipping build: **four clean, one failure** —
`ADD(type=f16,...)` with error `1.04e-7` against a `1.0e-7` tolerance. This is the upstream
tolerance defect, it reproduces on the unpatched base at a similar rate, and it is tracked as
`UPSTREAM-ADD-F16-TOL`. Anything reproducible, or far over tolerance, is not this.

---

## Limitations

**IQ-quantised models gain almost nothing from this stack.** Every performance patch here is
keyed to specific types — `#25940` and the MMVQ crossover cover K-quants, `#23685` covers
`Q4_K_M`, `#28398` covers `Q1_0`. `IQ4_XS` matches none of them and takes the identical code
path on base and v3:

| model | quant | v3 tg128 |
|---|---|---|
| Qwen3-4B Q4_K_M | K-quant | +34.0% |
| OLMoE-1B-7B | K-quant | +54.1% |
| gemma-26B IQ4_XS | IQ | +3.5% |
| Qwen3.8-27B IQ4_XS | IQ | +5.4% |

The two IQ models are the two flat ones. Extending the RDNA4 threshold tables to IQ types is
the obvious next target.

**`Q5_1` is unmeasured on gfx1201.** It shares a `case` with `Q6_K` upstream; we split it back
out and left it at the upstream threshold rather than ship an unverified change. No benchmark
model uses it.

**`rwkv7-1.5B` prefill is intrinsically noisy** — 13–15% min-to-max across 5 repetitions even
on a completely idle machine. Treat its pp512 figures as indicative only.

**Same as v1 and v2:** one card, one OS, one driver — RX 9070 (gfx1201), Windows 11,
Adrenalin 32.0.31041.1004, ROCm 7.2. Untested on 9070 XT, 9060, R9700 or Linux. Seven models.
Not an exhaustive quality suite.

---

## Building

```
git checkout 73a43d1
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

cmake -S . -B build-hip -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang++.exe" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF
cmake --build build-hip --target llama-bench llama-batched-bench llama-perplexity test-backend-ops
```

That should touch **43 tracked files**. If it does not, a patch failed — check rather than
assume; `git apply` failures are easy to swallow. The configure step must run inside a
`vcvars64.bat` environment or the compiler probe fails.

### Required environment

```
set GGML_CUDA_DQ_MMV=1     # without it you lose roughly a third of the decode gain
set GGML_CUDA_DQ_Q6K=1
set HIP_VISIBLE_DEVICES=0
set ROCR_VISIBLE_DEVICES=0
```

### Verifying

```
build-hip\bin\test-backend-ops.exe
```

Expect `Backend ROCm0: OK` and **15738/15738**. Run it at least five times; see the note on
`ADD(f16)` above.

---

## Three changes that are ours, not upstream

**Q6_K always-MMQ on RDNA4** — described above. Corrects `#25940` for gfx1201.

**RDNA4-MMVQ-XOVER** — `ggml_cuda_should_use_mmvq()` carries measured per-architecture tables
for Ada, Blackwell, GB10, Orin, CDNA1 and CDNA2 but **none for RDNA**, so RDNA4 fell through to
a permissive `ne11 <= 8`. Measurement shows MMQ beats MMVQ for K-quants once `ne11` reaches 4,
so the table returns `ne11 <= 3` for `Q4_K` and `Q6_K`.

**`#23685`, hand-ported** — the upstream patch predates template parameters the base has since
gained (`small_k`, `halve_iters`); `mmvq.cu` rejected 7 hunks. The port threads those through
the policy struct. Two deliberate divergences from the PR: it keeps `ggml_cuda_kernel_launch`
where the PR reverts to raw `<<<>>>` (which would strip PDL), and it restores
`ggml_cuda_pdl_sync()`.

Every other optimisation here was written by llama.cpp contributors. This project contributes
measurement, porting and evidence. Links go to the original PRs; please credit their authors.

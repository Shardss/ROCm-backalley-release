# ROCm backalley release

Validating **stalled llama.cpp optimisation PRs** on real RDNA4 hardware, and shipping the ones that survive.

llama.cpp has ~1,200 open pull requests. Many AMD-specific optimisations sit unreviewed for months — not because they are bad, but because **nobody with the hardware measures them**. A maintainer cannot merge an RDNA4 tuning change without someone running it on RDNA4.

This project does that measuring, keeps what works, and publishes both the evidence and a ready-to-run build.

---

## Latest release: v4

**v4 is a long-context release.** Flash attention on this card was spilling registers inside the
loop that walks the KV cache, so the cost grew with every token of context. The compiler reports it
directly:

| kernel | v3 | v4 |
|---|---|---|
| head_dim 128 — most models | 210 spills | **37** |
| head_dim 256 — Qwen3.8-27B | *kernel disabled on AMD* | **149** |

Prefill, measured from the packaged binaries against the shipped v3:

| model | at 8k context | at 16k context |
|---|---|---|
| Qwen3-4B Q4_K_M | **+14.2%** | **+22.5%** |
| Qwen3-8B Q4_K_M | **+9.0%** | — |
| Qwen3.8-27B IQ4_XS | **+13.3%** | **+20.3%** |

Plus **+6.0% decode on gemma-26B IQ4_XS**, from extending #23685's packed activation layout to the
IQ types it never covered.

**The gain grows with context depth and is near zero on an empty cache** — that is the signature of
the bug, which lived inside the KV loop. If you run short prompts, v4 gives you almost nothing.

**One caveat, stated up front: v4 is not bit-identical to upstream and v3 was.** Perplexity moves
+0.115% and ~2.2% of tokens pick a different top token than the base. That is the flash-attention
changes altering accumulation order. The quality gate passes. If you need output identical to
upstream, stay on v3.

Full detail — the retune that makes PR #26419 worth anything, all measurements, the quality gate,
and the VRAM arithmetic that decides how much context actually fits:
**[RELEASE_NOTES_v4.md](RELEASE_NOTES_v4.md)**

---

## AI agent disclosure

**This project was performed with frequent help from an AI agent.**

All *numbers* are machine-produced by `llama-bench`, `llama-perplexity` and `test-backend-ops`, executed by the agent. Direction and the final calls are the repository owner's: which candidates to chase, how far to push a branch, when an answer was not good enough, and what ships. Several findings here exist only because the owner rejected the agent's first conclusion and sent it back to measure again — including results the agent had written off.

| Step | Who |
|---|---|
| Candidate selection from ~1,200 open PRs | agent read titles/diffs and proposed; owner chose |
| Deciding whether a PR is "still valid" | agent read the code; owner arbitrated |
| **Porting #24386 and #26301 to the frozen base** | **agent wrote the merged code** |
| Triage verdicts (superseded / not-applicable / CUDA-only) | agent |
| Which branches to pursue, and how far | owner |
| Stacking order and what to exclude | owner, on the agent's measurements |
| Harness scripts, builds, benchmark runs | agent |
| Interpreting results and drafting this README | agent drafted; owner challenged and corrected |
| What ships in a release | owner |

Two ports (#23685, #26419) were **deliberately abandoned** at v3 rather than guessed at — see *Not ported*. Both were completed later, in v4.

The human owner of this repository is responsible for its contents.

---

## Hardware and base

| | |
|---|---|
| GPU | AMD Radeon RX 9070 (**gfx1201**, RDNA4), 16 GB |
| Driver | Adrenalin 32.0.31041.1004 (2026-08-17) |
| CPU / RAM | Ryzen 9 9900X, 128 GB DDR5 |
| OS | Windows 11 |
| Toolchain | ROCm HIP SDK 7.2, MSVC 14.44 (VS2022) |
| **Frozen base** | llama.cpp **`73a43d1`** — tagged `backalley-base`, never moves |

> MSVC 14.51 (VS2026) **fails** to build ROCm 7.2 HIP with `__clang_cuda_math_forward_declares.h` errors. Use VS2022. See upstream PR #24929.

Scope: **HIP backend only**. Vulkan is out of scope for v1.

---

## Method

1. **Freeze a base.** One commit, `73a43d1`. Every measurement is against it. It does not move until a release is cut.
2. **Relevance check** *(manual, AI agent)*. Is the PR's idea already upstream in our base? A failing `git apply` is **not** a rejection — it usually means the surrounding code moved and the patch needs porting.
3. **Build once, reuse.** Baseline binaries are built once and cached; only the patched build is rebuilt per candidate.
4. **A/B measurement with variance gates.** Baseline and patched are run as *separate process invocations*, not just `-r` repetitions. A result counts only if:
   - the min/max ranges are **disjoint**, and
   - the effect is **≥1%** (sub-1% differences between two builds are build-to-build noise, not results)
   - overlapping metrics **auto-escalate** to more runs before any verdict
5. **Verify the build actually changed.** The patched `ggml-hip.dll` must differ from baseline by content hash, and reverting must reproduce the baseline DLL byte-for-byte.
6. **Pin the GPU.** `HIP_VISIBLE_DEVICES=0` + `ROCR_VISIBLE_DEVICES=0`, and every run asserts it ran on the RX 9070. The machine also has an iGPU and an RTX 5060 Ti.
7. **Stack cumulatively.** Patches are applied one at a time on top of each other, measuring after each. A conflict or build failure drops only that patch.
8. **Quality gate the result.** See below.

### Models used

| model | why |
|---|---|
| Qwen3-4B Q4_K_M | dense, K-quant |
| Qwen3-8B Q4_K_M | dense, larger |
| Qwen3-4B Q4_0 | non-K-quant control |
| Qwen3-4B IQ4_XS | IQ-quant control |
| **OLMoE-1B-7B Q4_K_M** | **MoE — added mid-project, and it immediately caught a 47% regression** |

---

## Release v1

**Stack: #26301 + #28102 + #24386 + #25940** — branch `backalley-v1`, four commits on top of `73a43d1`.

### Cumulative gain vs the frozen base

| model | prompt processing (pp512) | token generation (tg128) |
|---|---|---|
| Qwen3-4B Q4_K_M | **~+11.8%** | **+26.3%** |
| Qwen3-8B Q4_K_M | **~+12.0%** | +10.0% |
| OLMoE-1B-7B (MoE) | ~+6.3% | **+35.4%** |
| Qwen3-4B Q4_0 | +2.4% | +2.9% |
| Qwen3-4B IQ4_XS | +2.0% | +4.7% |

No regressions on any model or metric.

> **#26301 requires `GGML_CUDA_DQ_MMV=1`.** Its upstream `arch_default` is RDNA3.5 and *excludes RDNA4*, so on a 9070 the path is compiled but never taken unless you set it. Roughly a third of the decode gain depends on this.

### What survived the filter

| PR | title | individual gain on gfx1201 |
|---|---|---|
| [#24386](https://github.com/ggml-org/llama.cpp/pull/24386) | tune RDNA4 MMVQ warps for K-quants | tg128 **+20.3%** dense Q4_K_M, **+35.4%** MoE |
| [#26301](https://github.com/ggml-org/llama.cpp/pull/26301) | dequant-float matvec (mmvdq) for Q4_K/Q5_K/Q6_K | tg128 **+9.7 / +7.4 / +4.5 / +4.3 / +2.9%** — *all five models* |
| [#25940](https://github.com/ggml-org/llama.cpp/pull/25940) | HIP RDNA4 MUL_MAT optimizations | pp512 **+10.4 / +10.1%** dense, **+6.5%** MoE |
| [#28102](https://github.com/ggml-org/llama.cpp/pull/28102) | CUDA/HIP Flash Attention tuning (gfx1201) | pp512 **+1.4 to +3.1%**, all quants |

Also validated but **not** in the stack:

| PR | why not |
|---|---|
| [#28398](https://github.com/ggml-org/llama.cpp/pull/28398) | `v_perm_b32` Q1_0 vec_dot — **tg128 +77.2%**, but only affects Q1_0, a quant almost nobody ships |

### Excluded, with evidence

| PR | verdict | reason |
|---|---|---|
| [#18816](https://github.com/ggml-org/llama.cpp/pull/18816) | **CONDITIONAL — excluded** | +15.6% dense prefill, but **−47.8% MoE prefill** (OLMoE pp512 10,115–10,141 → 5,168–5,370, n=3, disjoint ranges). Its RDNA4 `ne11<=256` thresholds misroute MoE expert matmuls. Needs gating on expert count. |
| [#20831](https://github.com/ggml-org/llama.cpp/pull/20831) | SUPERSEDED | Base already solves narrow-matrix warp waste via `calc_nwarps(..., small_k, halve_iters)`. Porting would replace a working mechanism with an older competing one. |
| [#21698](https://github.com/ggml-org/llama.cpp/pull/21698) | NOT APPLICABLE | RDNA2/GCN5 only; RDNA4 takes the unchanged `else` branch. |
| [#21170](https://github.com/ggml-org/llama.cpp/pull/21170) | CORRECTNESS ONLY | Removes a set-device early-return. Multi-GPU fix; no single-GPU performance benefit. |
| [#26487](https://github.com/ggml-org/llama.cpp/pull/26487) | CUDA ONLY | Does not compile under HIP — `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` is an undeclared identifier. |
| [#21849](https://github.com/ggml-org/llama.cpp/pull/21849) | DEFERRED | Tunable tile-selection infrastructure, but its CDNA entries are placeholders and it contains no RDNA4 values to gain. |
| 8 PRs | NO EFFECT | Build and run correctly; no measurable change on our models. |
| 6 PRs | OTHER ARCH | RDNA2/RDNA3/RDNA3.5/GCN/CDNA2 — can only be checked for non-regression. |
| 2 PRs | NEGLIGIBLE | Disjoint ranges but <1% on an unrelated path — build-to-build offset, not a real gain. |

### Not ported (deliberately)

| PR | why we stopped |
|---|---|
| [#23685](https://github.com/ggml-org/llama.cpp/pull/23685) | `vecdotq.cuh`/`quantize.cu` apply clean, but `mmvq.cu` rejects 7 hunks (~277 lines) including a 118-line policy template and a 122-line dispatch rewrite. The base gained `small_k`/`halve_iters` template parameters the PR predates. Design-level merge. |
| [#26419](https://github.com/ggml-org/llama.cpp/pull/26419) | 9/12 hunks apply; the 3 that matter conflict because the base refactored to `ggml_cuda_fattn_smem_swizzle::load_ldmatrix` (swizzled LDS) while the PR **bypasses** LDS for `DKQ > 128`. Grafting one onto the other is a semantic conflict in an output-critical kernel. |

In both cases a patch could be produced that *compiles and benchmarks fine while being quietly wrong*. That is worse than not porting. **An earlier auto-generated patch for #26419 was discarded for exactly this reason** — it applied cleanly only because it omitted the three hunks that carried the optimisation.

### Quality gate — no lobotomisation

Speed is worthless if the model gets dumber. Frozen base vs golden stack, Qwen3-4B Q4_K_M, wikitext (32 chunks), winogrande (150 tasks):

| check | base | golden stack | fail threshold | result |
|---|---|---|---|---|
| failing ops vs CPU reference | 0 | 0 | any *new* failure | **withdrawn — see below** |
| perplexity | 9.1136 | 9.1122 (**−0.0154%**) | >0.5% drift | **pass** |
| **mean KL divergence** | — | **0.002916 ± 0.000076** | >0.01 | **pass** |
| winogrande accuracy | 67.3333 | 67.3333 (**0 pts**) | −2 pts | **pass** |

Same-top-token agreement: **97.659% ± 0.167**. Median KLD 0.001297.

KL divergence compares the model's full output *distributions* against saved reference logits — it does not care that wording changed, only whether the model's beliefs moved. The residual is consistent with floating-point reduction-order changes, not altered behaviour.

> ### ⚠ Correction (2026-09-09): the ops row above is withdrawn
>
> Two faults were found in how that row was produced, and both are worth stating plainly because they
> affected every "0 failing ops" verdict this project published before this date.
>
> **The checker was wrong.** It counted only lines matching `OP(...): FAIL` and never read the process
> exit code or the `Backend <name>: OK|FAIL` summary. `test-backend-ops` does not mark a backend-level
> failure on each case — it prints `Failing tests:` and then lists the failing descriptors bare, which
> that checker ignored. A failing build could therefore report zero failing ops.
>
> **One run proves nothing.** `init_tensor_uniform()` seeds a per-thread `std::default_random_engine`
> from `std::random_device`, so **every run uses different input data**. On this hardware the
> **unpatched** frozen base fails `test-backend-ops` in **4 of 20 full runs**, on an f16 `ADD` case whose
> error lands 1.00–1.07e-07 against a 1.0e-07 tolerance. A single-run gate therefore rejects stock
> upstream llama.cpp about 20% of the time — pass or fail, it was close to a coin toss.
>
> **The perplexity, KL divergence and winogrande rows above still stand.** They were measured correctly
> and are unaffected. Only the ops row is withdrawn.
>
> The consequence was real: v1 shipped with a flash-attention regression this gate did not catch. It is
> fixed in v2, and v1 is left as published on purpose — see **Release v2** below and
> `RELEASE_NOTES_v2.md`. The corrected checker is `scripts/opscheck.ps1`.

**The stack is 12–26% faster and measurably the same model.**

---

## Release v2

**v2 removes one patch from v1 and adds nine.** The removal matters more than the additions: **#28102
broke flash attention** on split key/value head dimensions, and v1's single-run gate did not catch it.

**Carried over from v1:** [#26301](https://github.com/ggml-org/llama.cpp/pull/26301), [#24386](https://github.com/ggml-org/llama.cpp/pull/24386), [#25940](https://github.com/ggml-org/llama.cpp/pull/25940) — minus [#28102](https://github.com/ggml-org/llama.cpp/pull/28102), dropped.

### What v2 adds

| PR | title | individual gain on gfx1201 |
|---|---|---|
| [#23685](https://github.com/ggml-org/llama.cpp/pull/23685) | 4x packed Q8_1 activation for Q4_K_M in MMVQ (+Q5_K/Q6_K) | tg128 **+31.9%** Qwen3-4B, **+17.3%** Qwen3-8B, **+54.1%** OLMoE MoE |
| [#27248](https://github.com/ggml-org/llama.cpp/pull/27248) | CUDA support for q4_1, iq4_nl, q5_0 and q5_1 KV-cache types | quantized-KV decode **+167.9 / +163.7 / +164.8 / +147.5%** |
| [#27269](https://github.com/ggml-org/llama.cpp/pull/27269) | enable q8_0-K / q4_0-V flash attention vector kernels | quantized-KV decode q8_0/q4_0 **+176.5%** (42.35 → 117.11 t/s) |
| [#28552](https://github.com/ggml-org/llama.cpp/pull/28552) | size routed MoE MMQ N-tiles from typical expert width | pp512 **+14.6%** OLMoE, **+10.6%** gemma-26B-A4B |
| [#28398](https://github.com/ggml-org/llama.cpp/pull/28398) | hardware `v_perm_b32` for Q1_0 vec_dot on AMD | tg128 **+77.2%** on Q1_0 (143.6 → 254.6 t/s) |
| [#25206](https://github.com/ggml-org/llama.cpp/pull/25206) | optimize RWKV7 inference by fusing graph operators | rwkv7-1.5B pp512 **+9.5%**, tg128 **+7.7%** |
| `RDNA4-MMVQ-XOVER` | *own experiment* — add the missing RDNA4 branch to `ggml_cuda_should_use_mmvq` | batched decode B8 **+67.6%** Qwen3-4B, **+104.8%** Qwen3-8B; B1/B2 and prefill flat |
| [#26504](https://github.com/ggml-org/llama.cpp/pull/26504) | non-contiguous tensors in CEIL op | no speed change; +5 ops out of `NOT SUPPORTED` |
| [#28477](https://github.com/ggml-org/llama.cpp/pull/28477) | strided ABS for F16 and F32 | no speed change; +13 ops out of `NOT SUPPORTED` |

The last two ship for **coverage, not speed** — they move shapes off the CPU fallback path.

Apply order and build instructions: **`RELEASE_NOTES_v2.md`**.

### Gain vs the frozen base

| model | pp512 | tg128 |
|---|---|---|
| Qwen3-4B Q4_K_M | +8.1% | **+32.9%** |
| Qwen3-8B Q4_K_M | +7.2% | +16.5% |
| OLMoE-1B-7B (MoE) | **+19.3%** | **+54.4%** |
| gemma-26B-A4B IQ4_XS | +8.0% | +2.6% |
| rwkv7-1.5B | +11.1% | +34.1% |
| Qwen3-4B Q1_0 | **−2.8%** | **+83.2%** |

**Quantized KV cache — the largest single win in this release:**

| K/V type | base tg32 | v2 tg32 | gain |
|---|---|---|---|
| q4_1 / q4_1 | 42.86 | 147.57 | **+244.3%** |
| q5_0 / q5_0 | 42.50 | 145.70 | **+242.8%** |
| q5_1 / q5_1 | 42.70 | 146.10 | **+242.2%** |
| iq4_nl / iq4_nl | 40.18 | 121.00 | **+201.1%** |
| q8_0 / q4_0 | 43.02 | 153.42 | **+256.6%** |

Five of eight quantized-KV configurations were running roughly 3× slower than they should. #27248 and
#27269 restore them. If you run a quantized KV cache, this is the reason to take v2.

Batched decode, Qwen3-4B Q4_K_M, 128 prompt / 64 generate: **693.8 t/s at 8 parallel sequences.**

Op coverage: **OK 14726** vs the base's 14692, with `NOT SUPPORTED` down from 7579 to 7568 — v2 runs
more shapes than the base, it does not merely run them faster.

### Quality gate — two legs, both passed

v1's gate only ever exercised **batched** work: an `ne11` probe showed **1 of 8282** mul_mat dispatches
ran at decode width. That is how a flash-attention regression walked through it. v2 adds a decode leg
(`-b 512 -ub 1`), where the same probe reads **7227 of 7227**.

| check | base | v2 | threshold | result |
|---|---|---|---|---|
| perplexity, batched | 9.1136 | 9.1194 (+0.0636%) | >0.5% | **pass** |
| perplexity, decode | 12.4893 | 12.4687 (−0.1649%) | >0.5% | **pass** |
| mean KLD, batched | — | 0.002894 (same-top 97.929%) | >0.01 | **pass** |
| mean KLD, decode | — | 0.00199 (same-top 98.333%) | >0.01 | **pass** |

Every perplexity figure was measured twice and reproduced to four decimal places.

### Correctness, measured against the base rather than assumed

| | full runs failed | families |
|---|---|---|
| **v2 stack** | **3 of 20** | ADD_ADD ×2, ADD ×1 |
| frozen base, unpatched | 4 of 20 | ADD_ADD ×2, ADD ×2, MUL_MAT ×1 |

**v2 fails less often than unpatched upstream.** Every failure on both sides is a marginal-tolerance
case: the f16 `ADD` family at 1.00–1.07e-07 against 1.0e-07, and one `MUL_MAT(q5_1, n=1)` on the
*base* at 0.000509 against 0.000500.

One `MUL_MAT_ID(q4_K, n=1)` failure was seen during gating and never recurred — 0 in 28 later full runs
and 0 in 40 filtered runs. The base's own marginal `MUL_MAT(q5_1, n=1)` failure is the same class of
event: sibling op, same decode width, quantized type, a margin a hair over tolerance, with no patches
applied. **Stated honestly: the gate logged that case string but not its error margin, so this is strong
supporting evidence, not proof.** Compare #28102, which missed by 1.6–8×, reproduced readily, and was
absent from the base — that is what a real fault looks like.

### Why #28102 was dropped

It breaks four `FLASH_ATTN_EXT` shapes at `hsk=192, hsv=128, kv_view=1` — split key/value head
dimensions, as used by DeepSeek-style MLA attention — with errors of 0.000815, 0.001260, 0.002247 and
0.003965 against a 0.000500 tolerance. **The frozen base runs all four of those shapes and reports OK**,
so this is a regression, not newly exposed coverage. Isolated by removing each v1 patch in turn, 12 runs
per configuration: dropping #26301, #24386 or #25940 leaves the failures; dropping #28102 removes them.

None of the benchmark models use that attention shape, which is exactly why the quality gate passed it.

The cost of removal is honest and paid in prefill: roughly −4% on most models, −7.3% on gemma-26B.
Decode and quantized-KV gains are untouched. For a release other people run, a silent accuracy
regression on a real architecture's attention path is not worth 4% of prefill.

**v1 is deliberately left as published.** It has already been distributed; the correction ships as v2
rather than by rewriting a release people may already be using. If you are running v1 with an
MLA-architecture model, move to v2.

### Also rejected

| candidate | verdict |
|---|---|
| DPP warp reduction (own experiment) | Fails `test-backend-ops` deterministically — 3 of 3 runs, breaking `GATED_DELTA_NET`, `MUL_MAT_VEC_FUSION` and `TOPK_MOE`, with passing cases collapsing from ~14697 to ~12480. Its measured gain was +0.5–1.2%, at or below the ~0.95% noise floor. It had been cleared twice by the defective checker described above. |

---

## Release v3

**v3 fixes a prefill regression that shipped in both v1 and v2.**

[#25940](https://github.com/ggml-org/llama.cpp/pull/25940)'s RDNA4 table routes `Q6_K` matmuls to hipBLAS above `ne11 = 256`. On gfx1201 that path
is 2.6x to 6.7x slower than MMQ, so batched prefill lost up to **85%** against the *unpatched*
base — in every release to date. Neither previous quality gate measured the shape it lived in.

`Qwen3-4B Q4_K_M`, `-fa 1`, prefill throughput (tok/s):

| ne11 | base | v2 (shipped) | v3 |
|---|---|---|---|
| 384 | 5,376.65 | **793.92** (−85.2%) | 5,767.70 (+7.3%) |
| 512 | 5,944.91 | **2,352.14** (−60.4%) | 6,366.96 (+7.1%) |
| 640 | 5,433.94 | **1,204.93** (−77.8%) | 5,785.50 (+6.5%) |
| 1024 | 6,190.17 | **3,555.84** (−42.6%) | 6,583.93 (+6.4%) |

**If you run parallel sequences — a server, an agent, batched work — upgrade.** If you only ever
run one prompt at a time, v2 was about 4 points faster on single-sequence prefill, and v3 gives
that up deliberately to close the hole.

v3 also retires one of our three non-upstream patches: upstream [#28079](https://github.com/ggml-org/llama.cpp/pull/28079) fixed the root cause our
`cand_27269_with_hipfix.diff` was working around.

Full detail — seven models, quantized-KV sweep, batched sweep, quality gate, and the limits
(IQ-quantised models gain almost nothing from this stack):
**[RELEASE_NOTES_v3.md](RELEASE_NOTES_v3.md)**

v1 and v2 are **left as published**. They have been distributed, and rewriting them is worse than
shipping a clear correction.

---

---

## Reproduce

```bash
git clone https://github.com/Shardss/ROCm-backalley-release
cd ROCm-backalley-release

# 1. get llama.cpp at the frozen base
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout 73a43d1 && git tag backalley-base

# 2. apply the stack, in this order (v4 - 18 patches)
for p in stack_26301 stack_24386 stack_25940 rel_25206 rel_27248 rel_26504 rel_28477 \
         rel_28079 rel_27269_new pr28552 pr23685_ported cand_RDNA4_MMVQ_XOVER pr28398 \
         fix_Q6K_always_mmq v4_fa_spill v4_packed_q8_1_layout pr26419 v4_hd256_retune; do
  git apply ../patches/$p.diff || { echo "FAILED: $p"; break; }
done

# 3. build (ROCm 7.2 HIP SDK + MSVC 14.44/VS2022)
cmake -S . -B build-hip -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang++.exe" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF
cmake --build build-hip --target llama-bench

# 4. run - note the env vars, without them you lose ~1/3 of the decode gain
set GGML_CUDA_DQ_MMV=1
set GGML_CUDA_DQ_Q6K=1
set HIP_VISIBLE_DEVICES=0
build-hip\bin\llama-bench.exe -m <model.gguf> -ngl 99 -p 512 -n 128 -r 3
```

Or **download the prebuilt archive from Releases** — self-contained, no ROCm install required.

### Re-running the validation yourself

`scripts/` contains the harness:

| script | what it does |
|---|---|
| `v5.ps1` | validate one PR: cached baseline, patched build, variance-gated A/B on pp512+tg128 |
| `runner.ps1` | walk a queue of PRs unattended, recording verdicts |
| `stack.ps1` | cumulative safe merge, measuring after each addition |
| `goldengate.ps1` | the quality gate (ops / perplexity / KL divergence / winogrande) |

---

## Database

`database.csv` — every candidate, its verdict, the numbers, and the reasoning. 101 entries.

| status | count |
|---|---|
| VALIDATED | 18 |
| NO_EFFECT | 13 |
| NOT_APPLICABLE | 12 |
| NO_REGRESSION_ONLY | 6 |
| REJECTED | 6 |
| RESOLVED | 6 |
| SHIPPED | 5 |
| COMPLETE | 4 |
| SUPERSEDED / CORRECTNESS_ONLY / DEFERRED / CONFIRM_ONLY / NEGLIGIBLE / PASSED / FIXED | 2–3 each |
| 17 further one-off verdicts | 1 each |

**Hit rate: roughly 1 in 5.** Of 101 candidates examined, 18 are in the v4 stack. Several were initially rejected by a mechanical `git apply` check and only recovered by reading the code — including #23685, which needed a hand port, and #27269, which needed a one-line HIP fix the upstream PR omits.

---

## Honest limitations

- **One card, one OS, one driver.** RX 9070 (gfx1201), Windows 11, Adrenalin 32.0.31041.1004, ROCm 7.2. Untested on 9070 XT, 9060, R9700, or Linux.
- **The prebuilt archive is gfx1201-only.** Tensile libraries are filtered to gfx1201 to keep it at 370 MB; it will not work on other AMD architectures.
- **Five models, 32-chunk perplexity, 150 winogrande tasks.** Enough to catch a broken model; not an exhaustive quality suite.
- **Eighteen patches, not a hundred.** Out of 100 candidates examined.
- Upstream PR numbers, titles and author claims are reproduced in good faith; where our numbers disagree with an author's, both are shown.

## Credit

Every optimisation here was written by llama.cpp contributors, not by this project. This repo contributes **measurement, porting, and evidence** — nothing more. Links go to the original PRs; please credit their authors.

## Licence

The patches are derived from llama.cpp and carry its MIT licence. Scripts and documentation in this repo are MIT.

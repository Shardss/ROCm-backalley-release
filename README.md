# ROCm backalley release

Validating **stalled llama.cpp optimisation PRs** on real RDNA4 hardware, and shipping the ones that survive.

llama.cpp has ~1,200 open pull requests. Many AMD-specific optimisations sit unreviewed for months — not because they are bad, but because **nobody with the hardware measures them**. A maintainer cannot merge an RDNA4 tuning change without someone running it on RDNA4.

This project does that measuring, keeps what works, and publishes both the evidence and a ready-to-run build.

---

## ⚠ AI agent disclosure

**Every step in this project was performed by an AI agent (Claude, via Claude Code), driving a Windows machine over SSH.**

All *numbers* are machine-produced by `llama-bench`, `llama-perplexity` and `test-backend-ops`. All *judgement* is the agent's. Specifically, these steps were done manually by the AI agent and are not automated:

| Step | Done manually by an AI agent (Claude) |
|---|---|
| Candidate selection from ~1,200 open PRs | ✅ read titles/diffs, decided relevance to gfx1201 |
| Deciding whether a PR is "still valid" | ✅ read the code, checked whether upstream already merged the idea |
| **Porting #24386 and #26301 to the frozen base** | ✅ **the merged code was written by the agent, not the PR authors** |
| Triage verdicts (superseded / not-applicable / CUDA-only) | ✅ |
| Deciding the stacking order and what to exclude | ✅ |
| Writing all harness scripts | ✅ |
| Interpreting results and writing this README | ✅ |

Two ports (#23685, #26419) were **deliberately abandoned** rather than guessed at — see *Not ported*.

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
| failing ops vs CPU reference | 0 | 0 | any *new* failure | **pass** |
| perplexity | 9.1136 | 9.1122 (**−0.0154%**) | >0.5% drift | **pass** |
| **mean KL divergence** | — | **0.002916 ± 0.000076** | >0.01 | **pass** |
| winogrande accuracy | 67.3333 | 67.3333 (**0 pts**) | −2 pts | **pass** |

Same-top-token agreement: **97.659% ± 0.167**. Median KLD 0.001297.

KL divergence compares the model's full output *distributions* against saved reference logits — it does not care that wording changed, only whether the model's beliefs moved. The residual is consistent with floating-point reduction-order changes, not altered behaviour.

**The stack is 12–26% faster and measurably the same model.**

---

## Reproduce

```bash
git clone https://github.com/Shardss/ROCm-backalley-release
cd ROCm-backalley-release

# 1. get llama.cpp at the frozen base
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout 73a43d1 && git tag backalley-base

# 2. apply the stack, in this order
git apply ../patches/stack_26301.diff
git apply ../patches/stack_28102.diff
git apply ../patches/stack_24386.diff
git apply ../patches/stack_25940.diff

# 3. build (ROCm 7.2 HIP SDK + MSVC 14.44/VS2022)
cmake -S . -B build-hip -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang++.exe" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF
cmake --build build-hip --target llama-bench

# 4. run - note the env var, without it you lose ~1/3 of the decode gain
set GGML_CUDA_DQ_MMV=1
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

`database.csv` — every candidate, its verdict, the numbers, and the reasoning. 30 entries.

| status | count |
|---|---|
| VALIDATED | 5 |
| NO_EFFECT | 8 |
| NO_REGRESSION_ONLY (other arch) | 6 |
| NEEDS_PORT_HARD | 2 |
| NEGLIGIBLE | 2 |
| CONDITIONAL / SUPERSEDED / CUDA_ONLY / NOT_APPLICABLE / CORRECTNESS_ONLY / DEFERRED / CONFIRM_ONLY | 7 |

**Hit rate: roughly 1 in 6.** Two of the four PRs in the stack were initially rejected by a mechanical `git apply` check and only recovered by reading the code.

---

## Honest limitations

- **One card, one OS, one driver.** RX 9070 (gfx1201), Windows 11, Adrenalin 32.0.31041.1004, ROCm 7.2. Untested on 9070 XT, 9060, R9700, or Linux.
- **The prebuilt archive is gfx1201-only.** Tensile libraries are filtered to gfx1201 to keep it at 370 MB; it will not work on other AMD architectures.
- **Five models, 32-chunk perplexity, 150 winogrande tasks.** Enough to catch a broken model; not an exhaustive quality suite.
- **Four patches, not twenty.** Out of 30 candidates.
- Upstream PR numbers, titles and author claims are reproduced in good faith; where our numbers disagree with an author's, both are shown.

## Credit

Every optimisation here was written by llama.cpp contributors, not by this project. This repo contributes **measurement, porting, and evidence** — nothing more. Links go to the original PRs; please credit their authors.

## Licence

The patches are derived from llama.cpp and carry its MIT licence. Scripts and documentation in this repo are MIT.

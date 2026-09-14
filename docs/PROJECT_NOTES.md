# rocm-backalley

Validating stalled llama.cpp optimisation PRs on real RDNA4 hardware.

## Frozen base
`73a43d1f69345aee8bb186ef4b3172cef892f2e5` — tagged `backalley-base` on the build box.
DO NOT MOVE without explicit instruction. origin fetch refspec is disabled to prevent drift.

## Hardware
AMD Radeon RX 9070 (gfx1201, RDNA4), Adrenalin 32.0.31041.1004
Ryzen 9 9900X, 128 GB DDR5, Windows 11
HIP SDK 7.2, built with MSVC 14.44 (VS2022). MSVC 14.51/VS2026 fails - see PR #24929.
Scope: HIP backend only for now. Vulkan parked.

## Method per entry
1. Is it still valid against the frozen base? (idea not already merged; changes make sense for us)
2. Apply (port by hand if it does not apply cleanly)
3. A/B/A benchmark, 5 reps, GPU pinned to the 9070, rebuild verified by DLL content hash
   - patched DLL must differ from upstream
   - revert must reproduce the byte-identical upstream DLL
4. Quality gate: op correctness vs CPU, perplexity delta <0.5%, KL divergence <0.01, winogrande drop <2pts
5. Post result upstream to the PR
6. Record here

## Status
See database.csv

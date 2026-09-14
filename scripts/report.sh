#!/bin/bash
# fetch runner results and print the current table
scp -i ~/.ssh/handwriting_win11 -o ConnectTimeout=20 -o BatchMode=yes \
  'i3lue@100.122.119.94:C:/Projects/moe-4gb-search/results/queue.csv' \
  ~/Documents/rocm-backalley/queue_snapshot.csv 2>/dev/null
python3 - <<'EOF'
import csv,os
p=os.path.expanduser('~/Documents/rocm-backalley/queue_snapshot.csv')
rows=list(csv.reader(open(p)))[1:]
titles={
 '26301':'dequant-float matvec Q4_K/Q5_K/Q6_K','18816':'HIP mmq/rocblas switching RDNA4',
 '28398':'v_perm_b32 Q1_0 vec_dot (+110% claim)','27962':'HIP IQ2/IQ3 SWAR',
 '28178':'HIP small D2D copy kernel','28450':'gemma4-26b-a4b FA shape tune',
 '21849':'tunable mul_mat_q tile selection','21698':'VRAM->LDS load pipeline q8_0',
 '25834':'DP2A fallback for DP4A','26592':'CUB path on HIP via hipCUB',
 '27936':'hipCUB DeviceReduce SUM/MEAN','28313':'ROCm TOP_K kernels',
 '25863':'avoid ROCm_Host compute on iGPU','21170':'fix ROCm multi-GPU illegal access',
 '26487':'GGML_CUDA_BLOCKING_SYNC','20831':'dynamic MMVQ nwarps narrow matrices',
 '28339':'derive MMVQ nwarps from launch bounds','28102':'Flash Attention tuning gfx1201',
 '26419':'MMA FA head dim 256 on AMD RDNA','28447':'RDNA4 optimizations for GDN'}
print("| # | PR | what | result | detail |")
print("|---|---|---|---|---|")
print("| — | #24386 | RDNA4 MMVQ warps for K-quants | **VALIDATED** | decode +20.3% Q4_K_M — posted |")
print("| — | #25940 | HIP RDNA4 MUL_MAT | **VALIDATED** | prefill +10.2/+10.6% Q4_K_M — posted |")
for i,r in enumerate(rows,1):
    if len(r)<3: continue
    pr,st,detail=r[0],r[1],r[2]
    if st=='IMPROVEMENT': st='**IMPROVEMENT**'
    d=detail.replace('|','/')[:60]
    print("| %d | #%s | %s | %s | %s |" % (i,pr,titles.get(pr,''),st,d))
print()
print("%d of 20 processed" % len(rows))
EOF

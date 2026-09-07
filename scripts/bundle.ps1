$out="C:\Projects\rocm-backalley-v1"
$rb="C:\Program Files\AMD\ROCm\7.2\bin"
$res="C:\Projects\moe-4gb-search\results"; $log="$res\bundle.log"
function L($m){ "$((Get-Date).ToString('HH:mm:ss'))  $m" | Out-File -Append -FilePath $log -Encoding utf8 }
"=== bundling a self-contained rocm-backalley-v1 $(Get-Date) ===" | Out-File -FilePath $log -Encoding utf8

# 1. the ROCm runtime DLLs the build actually links
$core=@("hipblas.dll","rocblas.dll","amdhip64_7.dll","amd_comgr.dll","libhipblaslt.dll","hipblaslt.dll")
foreach($d in $core){
  $src="$rb\$d"
  if(Test-Path $src){ Copy-Item $src $out -Force; L ("  + {0,-24} {1:N1} MB" -f $d, ((Get-Item $src).Length/1MB)) }
  else { L ("  - $d not found in ROCm bin") }
}
# 2. Tensile / rocBLAS kernel libraries - gfx1201 ONLY, to keep the size sane
foreach($libdir in @("rocblas\library","hipblaslt\library")){
  $src="$rb\$libdir"
  if(-not (Test-Path $src)){ L "  - $libdir absent"; continue }
  $dst="$out\$libdir"
  New-Item -ItemType Directory -Path $dst -Force | Out-Null
  $n=0; $bytes=0
  Get-ChildItem $src -File | Where-Object { $_.Name -notmatch "gfx" -or $_.Name -match "gfx1201|gfx120X" } | ForEach-Object {
    Copy-Item $_.FullName $dst -Force; $n++; $bytes+=$_.Length
  }
  L ("  + {0,-24} {1} files, {2:N0} MB (gfx1201 + arch-agnostic only)" -f $libdir,$n,($bytes/1MB))
}
# 3. VC++ runtime, so the target machine does not need the redistributable
$vc="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC"
$vcdir = Get-ChildItem $vc -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
if($vcdir){
  $crt = Join-Path $vcdir.FullName "x64\Microsoft.VC143.CRT"
  if(Test-Path $crt){
    Get-ChildItem $crt -Filter *.dll | ForEach-Object { Copy-Item $_.FullName $out -Force; L ("  + " + $_.Name) }
  }
}
if(-not (Test-Path "$out\MSVCP140.dll")){
  foreach($d in @("MSVCP140.dll","VCRUNTIME140.dll","VCRUNTIME140_1.dll")){
    if(Test-Path "C:\Windows\System32\$d"){ Copy-Item "C:\Windows\System32\$d" $out -Force; L ("  + $d (from System32)") }
  }
}
$tot = (Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum).Sum
L ("archive now: " + (Get-ChildItem $out -Recurse -File | Measure-Object).Count + " files, " + [math]::Round($tot/1GB,2) + " GB")

# 4. prove it is self-contained: run with a MINIMAL PATH (no ROCm on PATH)
L "--- self-containment test: PATH stripped of ROCm ---"
$saved=$env:PATH
$env:PATH="C:\Windows\System32;C:\Windows"
$env:HIP_VISIBLE_DEVICES="0"; $env:ROCR_VISIBLE_DEVICES="0"; $env:GGML_CUDA_DQ_MMV="1"
Remove-Item env:HIP_PATH -ErrorAction SilentlyContinue
$raw = & "$out\llama-bench.exe" -m "C:\Projects\moe-4gb-search\models\Qwen3-4B-Q4_K_M.gguf" -ngl 99 -p 512 -n 128 -r 2 -o csv 2>&1
$env:PATH=$saved
$pp=$null;$tg=$null
foreach($x in ($raw | Where-Object { $_ -match '^"' })){
  $c=$x -split ','; $np=($c[-8] -replace '"',''); $ng=($c[-7] -replace '"','')
  if($np -eq "512" -and $ng -eq "0"){ $pp=[math]::Round([double]($c[-2] -replace '"',''),2) }
  if($np -eq "0" -and $ng -eq "128"){ $tg=[math]::Round([double]($c[-2] -replace '"',''),2) }
}
if($tg){ L ("  SELF-CONTAINED: runs with no ROCm on PATH. pp512=$pp tg128=$tg") }
else { L ("  NOT self-contained: " + (($raw|Select-Object -Last 3) -join ' | ')) }
L "=== BUNDLE DONE ==="

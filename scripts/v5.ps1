# v5.ps1 -Pr <n> [-Fresh]
#   Baseline measurements are CACHED per session (baseline_cache.json) since the
#   frozen-base binaries never change. -Fresh forces re-measurement.
#   Escalation re-runs ONLY the metrics that overlapped.
param([Parameter(Mandatory=$true)][string]$Pr,[switch]$Fresh,[int]$ARuns=5,[int]$BRuns=3,[string]$PatchEnv="")
$repo="C:\Projects\lcpp"; $res="C:\Projects\moe-4gb-search\results"
$log="$res\v5_$Pr.log"; $bl="$repo\bin-baseline"; $bb="$repo\build-hip\bin"
$cache="$res\baseline_cache.json"
$models=@(
  @{n="Qwen3-4B-Q4_K_M";p="C:\Projects\moe-4gb-search\models\Qwen3-4B-Q4_K_M.gguf"},
  @{n="Qwen3-8B-Q4_K_M";p="C:\Projects\moe-4gb-search\models\Qwen3-8B-Q4_K_M.gguf"},
  @{n="Qwen3-4B-Q4_0";  p="C:\Projects\moe-4gb-search\models\qwen3-4b-test-Q4_0.gguf"},
  @{n="Qwen3-4B-IQ4_XS";p="C:\Projects\moe-4gb-search\models\qwen3-4b-test-IQ4_XS.gguf"},
  @{n="OLMoE-1B-7B-Q4_K_M";p="C:\Projects\moe-4gb-search\models\olmoe-q4_k_m.gguf"}
)
$metrics=@("pp512","tg128")
function L($m){ "$((Get-Date).ToString('HH:mm:ss'))  $m" | Out-File -Append -FilePath $log -Encoding utf8 }
function Die($m){ L ("ABORT: "+$m); L "=== V5 DONE ==="; exit 1 }
function One($dir,$mp){
  $env:HIP_VISIBLE_DEVICES="0"; $env:ROCR_VISIBLE_DEVICES="0"
  if($script:ApplyEnv){ foreach($kv in ($script:ApplyEnv -split ';')){ if($kv){ $p=$kv -split '='; Set-Item "env:$($p[0])" $p[1] } } }
  else { foreach($v in @("GGML_CUDA_DQ_MMV","GGML_CUDA_DQ_Q6K")){ Remove-Item "env:$v" -ErrorAction SilentlyContinue } }
  $raw = & "$dir\llama-bench.exe" -m $mp -ngl 99 -p 512 -n 128 -r 3 -o csv 2>&1
  $pp=$null;$tg=$null
  foreach($x in ($raw | Where-Object { $_ -match '^"' })){
    $c=$x -split ','; $np=($c[-8] -replace '"',''); $ng=($c[-7] -replace '"','')
    if($np -eq "512" -and $ng -eq "0"){ $pp=[double]($c[-2] -replace '"','') }
    if($np -eq "0" -and $ng -eq "128"){ $tg=[double]($c[-2] -replace '"','') }
  }
  if(($raw -join ' ') -notmatch "RX 9070"){ Die "not on the RX 9070" }
  if($tg -eq $null -or $pp -eq $null){ Die ("no result: " + (($raw|Select-Object -Last 2) -join ' | ')) }
  return @{pp=$pp;tg=$tg}
}
function RunBench($dir,$tag,$n,$acc,$only){
  for($i=1;$i -le $n;$i++){
    foreach($m in $models){
      if(-not (Test-Path $m.p)){ continue }
      if($only -and -not ($only | Where-Object { $_ -like "$($m.n)|*" })){ continue }
      $r = One $dir $m.p
      $acc["$($m.n)|pp512"] += $r.pp; $acc["$($m.n)|tg128"] += $r.tg
    }
  }
  return $acc
}
"=== PR #$Pr vs frozen base $(& git -C $repo rev-parse --short HEAD)  $(Get-Date) ===" | Out-File -FilePath $log -Encoding utf8
if(-not (Test-Path "$bl\llama-bench.exe")){ Die "no baseline snapshot" }
$env:HIP_PATH="C:\Program Files\AMD\ROCm\7.2\"; $env:PATH=$env:HIP_PATH+"bin;"+$env:PATH
$blSha=(Get-FileHash "$bl\ggml-hip.dll" -Algorithm SHA256).Hash.Substring(0,16)

# ---- cached baseline -------------------------------------------------------
$A=@{}
$useCache=$false
if((Test-Path $cache) -and -not $Fresh){
  $c = Get-Content $cache -Raw | ConvertFrom-Json
  if($c.sha -eq $blSha){
    foreach($p in $c.data.PSObject.Properties){ $A[$p.Name] = @($p.Value) }
    $useCache=$true
    L ("baseline: CACHED (sha $blSha, " + $c.runs + " runs, measured " + $c.when + ")")
  } else { L "baseline cache stale (different binaries) - remeasuring" }
}
if(-not $useCache){
  foreach($m in $models){ foreach($mt in $metrics){ $A["$($m.n)|$mt"]=@() } }
  L "--- BASELINE x$ARuns (measuring, will be cached) ---"
  $A = RunBench $bl "base" $ARuns $A $null
  $obj=[ordered]@{ sha=$blSha; runs=$ARuns; when=(Get-Date).ToString('s'); data=$A }
  $obj | ConvertTo-Json -Depth 4 | Out-File $cache -Encoding utf8
  L "baseline cached"
}
foreach($m in $models){ foreach($mt in $metrics){
  $k="$($m.n)|$mt"; if(-not $A[$k]){ continue }
  $v=$A[$k]; L ("    base   {0,-19} {1}  n={2} min={3,-9:N2} max={4,-9:N2}" -f $m.n,$mt,$v.Count,($v|Measure-Object -Minimum).Minimum,($v|Measure-Object -Maximum).Maximum)
} }
# ---- patch + build ---------------------------------------------------------
L "--- applying PR #$Pr ---"
& git -C $repo checkout -- . 2>&1 | Out-Null
$ap = & git -C $repo apply "C:\Users\I3lue\pr$Pr.diff" 2>&1
if($LASTEXITCODE -ne 0){ Die ("patch does not apply: " + ($ap -join ' ')) }
L ("changed: " + ((& git -C $repo diff --name-only) -join ', '))
& cmd /c "call `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set `"PATH=C:\Program Files\AMD\ROCm\7.2\bin;C:\Program Files\CMake\bin;C:\msys64\mingw64\bin;%PATH%`" && cmake --build $repo\build-hip --target llama-bench" 2>&1 | Out-File "$res\build_$Pr.log"
$tl=Get-Content "$res\build_$Pr.log" -Tail 3
if(($tl -join ' ') -match "build stopped|FAILED|permission denied"){ $tl|ForEach-Object{L ("   "+$_)}; Die "build failed" }
$hB=(Get-FileHash "$bb\ggml-hip.dll" -Algorithm SHA256).Hash.Substring(0,16)
if($hB -eq $blSha){ Die "patched DLL identical to baseline" }
L "  base dll=$blSha  patched dll=$hB"
# ---- patched ---------------------------------------------------------------
$B=@{}; foreach($m in $models){ foreach($mt in $metrics){ $B["$($m.n)|$mt"]=@() } }
if($PatchEnv){ L ("patched runs use env: " + $PatchEnv) }
$script:ApplyEnv = $PatchEnv
L "--- PATCHED x$BRuns ---"; $B = RunBench $bb "patched" $BRuns $B $null
$script:ApplyEnv = $null
function Verdict($lab){
  $ov=@()
  foreach($m in $models){ foreach($mt in $metrics){
    $k="$($m.n)|$mt"
    if(-not $A[$k] -or -not $B[$k] -or $B[$k].Count -eq 0){ continue }
    $amin=($A[$k]|Measure-Object -Minimum).Minimum; $amax=($A[$k]|Measure-Object -Maximum).Maximum
    $bmin=($B[$k]|Measure-Object -Minimum).Minimum; $bmax=($B[$k]|Measure-Object -Maximum).Maximum
    $am=($A[$k]|Measure-Object -Average).Average; $bm=($B[$k]|Measure-Object -Average).Average
    $g=[math]::Round(($bm-$am)/$am*100,2)
    if($bmin -gt $amax -and $g -ge 1.0){ $v="IMPROVEMENT" }
    elseif($bmax -lt $amin -and $g -le -1.0){ $v="REGRESSION" }
    elseif(($bmin -gt $amax) -or ($bmax -lt $amin)){ $v="NEGLIGIBLE($g%)" }
    else { $v="OVERLAP"; $ov += $k }
    L ("  $lab {0,-19} {1}  A=[{2:N2}..{3:N2}] B=[{4:N2}..{5:N2}] mean {6,7}% => {7}" -f $m.n,$mt,$amin,$amax,$bmin,$bmax,$g,$v)
  } }
  return $ov
}
$ov = Verdict " "
if($ov.Count -gt 0){
  L ("--- ESCALATE: only the " + $ov.Count + " overlapping metric(s) ---")
  $script:ApplyEnv = $null
  $A = RunBench $bl "base+" 3 $A $ov
  $script:ApplyEnv = $PatchEnv
  $B = RunBench $bb "patch+" 3 $B $ov
  $script:ApplyEnv = $null
  $null = Verdict "FINAL"
}
& git -C $repo checkout -- . 2>&1 | Out-Null
L "=== V5 DONE ==="

param([Parameter(Mandatory=$true)][string]$Pr,[int]$ARuns=5,[int]$BRuns=3)
$repo="C:\Projects\lcpp"; $res="C:\Projects\moe-4gb-search\results"
$log="$res\v4_$Pr.log"; $bl="$repo\bin-baseline"; $bb="$repo\build-hip\bin"
$models=@(
  @{n="Qwen3-4B-Q4_K_M";p="C:\Projects\moe-4gb-search\models\Qwen3-4B-Q4_K_M.gguf"},
  @{n="Qwen3-8B-Q4_K_M";p="C:\Projects\moe-4gb-search\models\Qwen3-8B-Q4_K_M.gguf"},
  @{n="Qwen3-4B-Q4_0";  p="C:\Projects\moe-4gb-search\models\qwen3-4b-test-Q4_0.gguf"},
  @{n="Qwen3-4B-IQ4_XS";p="C:\Projects\moe-4gb-search\models\qwen3-4b-test-IQ4_XS.gguf"}
)
$metrics=@("pp512","tg128")
function L($m){ "$((Get-Date).ToString('HH:mm:ss'))  $m" | Out-File -Append -FilePath $log -Encoding utf8 }
function Die($m){ L ("ABORT: "+$m); L "=== V4 DONE ==="; exit 1 }
function One($dir,$mp){
  $env:HIP_VISIBLE_DEVICES="0"; $env:ROCR_VISIBLE_DEVICES="0"
  $raw = & "$dir\llama-bench.exe" -m $mp -ngl 99 -p 512 -n 128 -r 3 -o csv 2>&1
  $pp=$null;$tg=$null
  foreach($x in ($raw | Where-Object { $_ -match '^"' })){
    $c=$x -split ','; $np=($c[-8] -replace '"',''); $ng=($c[-7] -replace '"','')
    if($np -eq "512" -and $ng -eq "0"){ $pp=[double]($c[-2] -replace '"','') }
    if($np -eq "0" -and $ng -eq "128"){ $tg=[double]($c[-2] -replace '"','') }
  }
  if(($raw -join ' ') -notmatch "RX 9070"){ Die "not on the RX 9070" }
  if($tg -eq $null -or $pp -eq $null){ Die ("no result: " + (($raw|Select-Object -Last 2) -join ' | ')) }
  return @{pp=$pp; tg=$tg}
}
function Runs($dir,$tag,$n,$acc){
  for($i=1;$i -le $n;$i++){
    foreach($m in $models){ $r = One $dir $m.p; $acc["$($m.n)|pp512"] += $r.pp; $acc["$($m.n)|tg128"] += $r.tg }
  }
  foreach($m in $models){ foreach($mt in $metrics){
    $v=$acc["$($m.n)|$mt"]; $mn=($v|Measure-Object -Minimum).Minimum; $mx=($v|Measure-Object -Maximum).Maximum; $av=($v|Measure-Object -Average).Average
    L ("    {0,-9} {1,-17} {2}  n={3}  min={4,-9:N2} max={5,-9:N2} mean={6,-9:N2} spread={7:N2}%" -f $tag,$m.n,$mt,$v.Count,$mn,$mx,$av,(($mx-$mn)/$av*100))
  } }
  return $acc
}
function Verdict($A,$B,$label){
  $ov=@()
  foreach($m in $models){ foreach($mt in $metrics){
    $k="$($m.n)|$mt"
    $amin=($A[$k]|Measure-Object -Minimum).Minimum; $amax=($A[$k]|Measure-Object -Maximum).Maximum
    $bmin=($B[$k]|Measure-Object -Minimum).Minimum; $bmax=($B[$k]|Measure-Object -Maximum).Maximum
    $am=($A[$k]|Measure-Object -Average).Average; $bm=($B[$k]|Measure-Object -Average).Average
    $g=[math]::Round(($bm-$am)/$am*100,2)
    # disjoint ranges AND a meaningful effect size - a sub-1% offset between two
    # builds is not a result, it is build-to-build variation
    $MINEFFECT = 1.0
    if($bmin -gt $amax -and $g -ge $MINEFFECT){ $v="IMPROVEMENT" }
    elseif($bmax -lt $amin -and $g -le (-1*$MINEFFECT)){ $v="REGRESSION" }
    elseif(($bmin -gt $amax) -or ($bmax -lt $amin)){ $v=("NEGLIGIBLE(" + $g + "%)") }
    else { $v="OVERLAP"; $ov += $k }
    L ("  $label {0,-17} {1}  A=[{2:N2}..{3:N2}] B=[{4:N2}..{5:N2}]  mean {6,7}%  => {7}" -f $m.n,$mt,$amin,$amax,$bmin,$bmax,$g,$v)
  } }
  return $ov
}
"=== PR #$Pr vs frozen base $(& git -C $repo rev-parse --short HEAD)  $(Get-Date) ===" | Out-File -FilePath $log -Encoding utf8
if(-not (Test-Path "$bl\llama-bench.exe")){ Die "no baseline snapshot" }
$env:HIP_PATH="C:\Program Files\AMD\ROCm\7.2\"; $env:PATH=$env:HIP_PATH+"bin;"+$env:PATH
$A=@{}; $B=@{}
foreach($m in $models){ foreach($mt in $metrics){ $A["$($m.n)|$mt"]=@(); $B["$($m.n)|$mt"]=@() } }
L "--- BASELINE x$ARuns ---"; $A = Runs $bl "base" $ARuns $A
L "--- applying PR #$Pr ---"
& git -C $repo checkout -- . 2>&1 | Out-Null
$ap = & git -C $repo apply "C:\Users\I3lue\pr$Pr.diff" 2>&1
if($LASTEXITCODE -ne 0){ Die ("patch does not apply: " + ($ap -join ' ')) }
L ("changed: " + ((& git -C $repo diff --name-only) -join ', '))
L "building B..."
& cmd /c "call `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set `"PATH=C:\Program Files\AMD\ROCm\7.2\bin;C:\Program Files\CMake\bin;C:\msys64\mingw64\bin;%PATH%`" && cmake --build $repo\build-hip --target llama-bench" 2>&1 | Out-File "$res\build_$Pr.log"
$tl=Get-Content "$res\build_$Pr.log" -Tail 3
if(($tl -join ' ') -match "build stopped|FAILED|permission denied"){ $tl|ForEach-Object{L ("   "+$_)}; Die "build failed" }
$hB=(Get-FileHash "$bb\ggml-hip.dll" -Algorithm SHA256).Hash.Substring(0,16)
$hA=(Get-FileHash "$bl\ggml-hip.dll" -Algorithm SHA256).Hash.Substring(0,16)
if($hB -eq $hA){ Die "patched DLL identical to baseline" }
L "  base dll=$hA  patched dll=$hB"
L "--- PATCHED x$BRuns ---"; $B = Runs $bb "patched" $BRuns $B
$ov = Verdict $A $B " "
if($ov.Count -gt 0){
  L ("--- ESCALATE +5 each (overlap on " + $ov.Count + " metric(s)) ---")
  $A = Runs $bl "base+" 5 $A
  $B = Runs $bb "patch+" 5 $B
  $ov2 = Verdict $A $B "FINAL"
}
& git -C $repo checkout -- . 2>&1 | Out-Null
L "tree reverted to frozen base"
L "=== V4 DONE ==="

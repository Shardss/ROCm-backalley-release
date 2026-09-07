# cumulative safe merge: add one patch at a time, verify apply+build, measure, keep or drop
$repo="C:\Projects\lcpp"; $res="C:\Projects\moe-4gb-search\results"
$log="$res\stack.log"; $bl="$repo\bin-baseline"; $bb="$repo\build-hip\bin"
function L($m){ "$((Get-Date).ToString('HH:mm:ss'))  $m" | Out-File -Append -FilePath $log -Encoding utf8 }
$models=@(
  @{n="Qwen3-4B-Q4_K_M";p="C:\Projects\moe-4gb-search\models\Qwen3-4B-Q4_K_M.gguf"},
  @{n="Qwen3-8B-Q4_K_M";p="C:\Projects\moe-4gb-search\models\Qwen3-8B-Q4_K_M.gguf"},
  @{n="Qwen3-4B-Q4_0";  p="C:\Projects\moe-4gb-search\models\qwen3-4b-test-Q4_0.gguf"},
  @{n="Qwen3-4B-IQ4_XS";p="C:\Projects\moe-4gb-search\models\qwen3-4b-test-IQ4_XS.gguf"},
  @{n="OLMoE-1B-7B";    p="C:\Projects\moe-4gb-search\models\olmoe-q4_k_m.gguf"}
)
# order: isolated first, then the shared-file ones
$order=@("26301","28102","24386","18816","25940")
$env:HIP_PATH="C:\Program Files\AMD\ROCm\7.2\"; $env:PATH=$env:HIP_PATH+"bin;"+$env:PATH
function Bench($dir,$tag,$envs){
  while((Get-Process llama-bench,test-backend-ops -ErrorAction SilentlyContinue|Measure-Object).Count -gt 0){ Start-Sleep -Seconds 8 }
  $env:HIP_VISIBLE_DEVICES="0"; $env:ROCR_VISIBLE_DEVICES="0"
  foreach($v in @("GGML_CUDA_DQ_MMV","GGML_CUDA_DQ_Q6K")){ Remove-Item "env:$v" -ErrorAction SilentlyContinue }
  if($envs){ foreach($kv in ($envs -split ';')){ if($kv){ $p=$kv -split '='; Set-Item "env:$($p[0])" $p[1] } } }
  $out=@{}
  foreach($m in $models){
    if(-not (Test-Path $m.p)){ continue }
    $raw = & "$dir\llama-bench.exe" -m $m.p -ngl 99 -p 512 -n 128 -r 3 -o csv 2>&1
    $pp=$null;$tg=$null
    foreach($x in ($raw | Where-Object { $_ -match '^"' })){
      $c=$x -split ','; $np=($c[-8] -replace '"',''); $ng=($c[-7] -replace '"','')
      if($np -eq "512" -and $ng -eq "0"){ $pp=[double]($c[-2] -replace '"','') }
      if($np -eq "0" -and $ng -eq "128"){ $tg=[double]($c[-2] -replace '"','') }
    }
    if(($raw -join ' ') -notmatch "RX 9070"){ L "  !!! $tag NOT ON 9070"; return $null }
    if($tg -eq $null){ L ("  !!! $tag $($m.n) no result"); return $null }
    $out[$m.n]=@{pp=[math]::Round($pp,2);tg=[math]::Round($tg,2)}
    L ("    {0,-10} {1,-17} pp512={2,-10} tg128={3}" -f $tag,$m.n,$out[$m.n].pp,$out[$m.n].tg)
  }
  return $out
}
"=== SAFE CUMULATIVE MERGE $(Get-Date) ===" | Out-File -FilePath $log -Encoding utf8
& git -C $repo checkout -- . 2>&1 | Out-Null
& git -C $repo clean -fd ggml/src/ggml-cuda 2>&1 | Out-Null
L "--- STAGE 0: frozen base ---"
$base = Bench $bl "base" $null
if(-not $base){ L "ABORT: baseline failed"; exit 1 }
$applied=@()
$envAcc=""
foreach($pr in $order){
  L "=========== adding #$pr on top of [$($applied -join '+')] ==========="
  $chk = & git -C $repo apply --check "C:\Users\I3lue\stack_$pr.diff" 2>&1
  if($LASTEXITCODE -ne 0){
    L ("  CONFLICT: cannot apply on top of the stack -> DROPPED. " + (($chk|Select-Object -First 1) -replace '"',''))
    continue
  }
  & git -C $repo apply "C:\Users\I3lue\stack_$pr.diff" 2>&1 | Out-Null
  L ("  applied. tree now: " + ((& git -C $repo diff --name-only) -join ', '))
  & cmd /c "call `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set `"PATH=C:\Program Files\AMD\ROCm\7.2\bin;C:\Program Files\CMake\bin;C:\msys64\mingw64\bin;%PATH%`" && cmake --build $repo\build-hip --target llama-bench" 2>&1 | Out-File "$res\stack_build_$pr.log"
  $tl = Get-Content "$res\stack_build_$pr.log" -Tail 3
  if(($tl -join ' ') -match "build stopped|FAILED|permission denied"){
    L "  BUILD FAILED with #$pr on the stack -> REVERTING this patch"
    & git -C $repo checkout -- . 2>&1 | Out-Null
    foreach($p in $applied){ & git -C $repo apply "C:\Users\I3lue\stack_$p.diff" 2>&1 | Out-Null }
    continue
  }
  if($pr -eq "26301"){ $envAcc = "GGML_CUDA_DQ_MMV=1;GGML_CUDA_DQ_Q6K=1" }
  $r = Bench $bb "with$pr" $envAcc
  if(-not $r){ L "  measurement failed -> reverting #$pr"; & git -C $repo checkout -- . 2>&1 | Out-Null; foreach($p in $applied){ & git -C $repo apply "C:\Users\I3lue\stack_$p.diff" 2>&1|Out-Null }; continue }
  $applied += $pr
  L "  --- cumulative vs frozen base ---"
  foreach($m in $models){
    if(-not $base[$m.n] -or -not $r[$m.n]){ continue }
    $dp=[math]::Round(($r[$m.n].pp-$base[$m.n].pp)/$base[$m.n].pp*100,2)
    $dt=[math]::Round(($r[$m.n].tg-$base[$m.n].tg)/$base[$m.n].tg*100,2)
    L ("    {0,-17} pp512 {1,7}%   tg128 {2,7}%" -f $m.n,$dp,$dt)
  }
}
L "=========== FINAL STACK: $($applied -join ' + ') ==========="
L "=== STACK DONE ==="

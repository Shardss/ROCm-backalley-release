# Build the golden stack, then run the full quality gate against the frozen base.
$repo="C:\Projects\lcpp"; $res="C:\Projects\moe-4gb-search\results"
$log="$res\goldengate.log"; $bl="$repo\bin-baseline"; $bb="$repo\build-hip\bin"
$model="C:\Projects\moe-4gb-search\models\Qwen3-4B-Q4_K_M.gguf"
$corpus="C:\Projects\moe-4gb-search\wiki.test.raw"
$wg="C:\Projects\moe-4gb-search\winogrande.csv"
$logits="$res\logits_ref.bin"
function L($m){ "$((Get-Date).ToString('HH:mm:ss'))  $m" | Out-File -Append -FilePath $log -Encoding utf8 }
function Idle(){ while((Get-Process llama-bench,test-backend-ops,llama-perplexity,llama-completion -ErrorAction SilentlyContinue|Measure-Object).Count -gt 0){ Start-Sleep -Seconds 8 } }
$env:HIP_PATH="C:\Program Files\AMD\ROCm\7.2\"; $env:PATH=$env:HIP_PATH+"bin;"+$env:PATH
$env:HIP_VISIBLE_DEVICES="0"; $env:ROCR_VISIBLE_DEVICES="0"
"=== QUALITY GATE: frozen base vs golden stack $(Get-Date) ===" | Out-File -FilePath $log -Encoding utf8

function OpsFail($dir,$tag){
  Idle
  $o = & "$dir\test-backend-ops.exe" 2>&1
  $t = ($o -join "`n") -replace "\x1b\[[0-9;]*m",""
  if($t -notmatch "RX 9070"){ L "  $tag : NOT ON 9070"; return $null }
  $f=@(); foreach($line in ($t -split "`n")){ if($line -match "^\s*(\S+\(.*?\)):\s*FAIL"){ $f += $Matches[1] } }
  L ("  $tag : failing ops = " + $f.Count)
  return $f
}
function Ppl($dir,$tag){
  Idle
  $p = & "$dir\llama-perplexity.exe" -m $model -f $corpus -ngl 99 --chunks 32 -c 512 2>&1
  $v=$null; foreach($l in $p){ if($l -match "Final estimate: PPL = ([\d.]+)"){ $v=[double]$Matches[1] } }
  L ("  $tag : PPL = $v")
  return $v
}
function Wino($dir,$tag){
  Idle
  if(-not (Test-Path $wg)){ return $null }
  $w = & "$dir\llama-perplexity.exe" -m $model -f $wg -ngl 99 --winogrande --winogrande-tasks 150 2>&1
  $a=$null
  foreach($l in $w){ if($l -match "Final Winogrande score\(\d+\s+tasks\):\s*([\d.]+)"){ $a=[double]$Matches[1] } }
  if($a -eq $null){ foreach($l in $w){ if($l -match "Winogrande.*?:\s*([\d.]+)\s*\+/-"){ $a=[double]$Matches[1] } } }
  L ("  $tag : winogrande = $a")
  return $a
}

L "--- FROZEN BASE ---"
$bf = OpsFail $bl "base"; $bp = Ppl $bl "base"; $bw = Wino $bl "base"
Idle
L "  base: saving reference logits for KL"
& "$bl\llama-perplexity.exe" -m $model -f $corpus -ngl 99 --chunks 32 -c 512 --kl-divergence-base $logits 2>&1 | Out-Null
if(Test-Path $logits){ L ("  logits saved: " + [math]::Round((Get-Item $logits).Length/1MB,1) + " MB") } else { L "  LOGITS NOT SAVED" }

L "--- building GOLDEN STACK (26301+28102+24386+25940) ---"
& git -C $repo checkout -- . 2>&1 | Out-Null
& git -C $repo clean -fd ggml/src/ggml-cuda 2>&1 | Out-Null
L "  tree cleaned (untracked files removed)"
foreach($p in @("26301","28102","24386","25940")){
  $c = & git -C $repo apply --check "C:\Users\I3lue\stack_$p.diff" 2>&1
  if($LASTEXITCODE -ne 0){ L ("  APPLY FAILED for #$p : " + (($c|Select-Object -First 1) -replace '"','')); exit 1 }
  & git -C $repo apply "C:\Users\I3lue\stack_$p.diff" 2>&1 | Out-Null
  L "  applied #$p"
}
& cmd /c "call `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set `"PATH=C:\Program Files\AMD\ROCm\7.2\bin;C:\Program Files\CMake\bin;C:\msys64\mingw64\bin;%PATH%`" && cmake --build $repo\build-hip --target llama-bench llama-perplexity llama-completion test-backend-ops" 2>&1 | Out-File "$res\goldengate_build.log"
$t=Get-Content "$res\goldengate_build.log" -Tail 3
if(($t -join ' ') -match "build stopped|FAILED|permission denied"){ L "BUILD FAILED"; exit 1 }
L "  golden stack built"

L "--- GOLDEN STACK (with GGML_CUDA_DQ_MMV=1) ---"
$env:GGML_CUDA_DQ_MMV="1"; $env:GGML_CUDA_DQ_Q6K="1"
$gf = OpsFail $bb "gold"; $gp = Ppl $bb "gold"; $gw = Wino $bb "gold"
Idle
L "  gold: KL divergence vs base reference"
$k = & "$bb\llama-perplexity.exe" -m $model -f $corpus -ngl 99 --chunks 32 -c 512 --kl-divergence --kl-divergence-base $logits 2>&1
$k | Out-File "$res\goldengate_kl.txt" -Encoding utf8
$kt=($k -join "`n")
$kl=$null; $top=$null
if($kt -match "Mean\s+KLD:\s*([\d.eE+-]+)"){ $kl=[double]$Matches[1] }
if($kt -match "Same top p:\s*([\d.]+)"){ $top=[double]$Matches[1] }
L ("  mean KLD = $kl   same-top-token = $top%")

L "=========== VERDICT ==========="
$new = @($gf | Where-Object { $bf -notcontains $_ })
L ("  newly failing ops : " + $new.Count + $(if($new.Count){" -> " + ($new -join ', ')}else{""}))
if($bp -and $gp){ $d=[math]::Round(($gp-$bp)/$bp*100,4); L ("  perplexity        : base $bp -> gold $gp  (" + $d + "%)") }
if($bw -ne $null -and $gw -ne $null){ L ("  winogrande        : base $bw -> gold $gw  (" + [math]::Round($gw-$bw,2) + " pts)") }
L ("  mean KLD          : $kl")
$fail=@()
if($new.Count -gt 0){ $fail += "$($new.Count) newly failing ops" }
if($bp -and $gp -and [math]::Abs(($gp-$bp)/$bp*100) -gt 0.5){ $fail += "perplexity moved >0.5%" }
if($kl -ne $null -and $kl -gt 0.01){ $fail += "KLD $kl > 0.01" }
if($bw -ne $null -and $gw -ne $null -and ($bw-$gw) -gt 2.0){ $fail += "winogrande dropped >2pts" }
if($fail.Count -gt 0){ L ("  RESULT: FAIL - " + ($fail -join '; ')) } else { L "  RESULT: PASS - golden stack is quality-neutral" }
& git -C $repo checkout -- . 2>&1 | Out-Null
& git -C $repo clean -fd ggml/src/ggml-cuda 2>&1 | Out-Null
L "=== GOLDENGATE DONE ==="

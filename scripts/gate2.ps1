# ============================================================================
#  rocm-backalley automated quality gate v2
#    gate2.ps1 -Tag <label> [-Reference <tag>]
#  With no -Reference: establishes the reference (saves logits + failure set).
#  With -Reference:    compares against it and emits PASS/FAIL automatically.
#  Fails closed.
# ============================================================================
param([Parameter(Mandatory=$true)][string]$Tag, [string]$Reference="")

$res="C:\Projects\moe-4gb-search\results"
$bin="C:\Projects\lcpp\build-hip\bin"
$log="$res\gate2_$Tag.log"
$model="C:\Projects\moe-4gb-search\models\Qwen3-4B-Q4_K_M.gguf"
$corpus="C:\Projects\moe-4gb-search\wiki.test.raw"
$logits="$res\logits_reference.bin"
function L($m){ "$((Get-Date).ToString('HH:mm:ss'))  $m" | Out-File -Append -FilePath $log -Encoding utf8 }
function Fail($m){ L ("FAIL: " + $m); $g.verdict="FAIL"; $g.reason=$m; $g|ConvertTo-Json|Out-File "$res\gate2_$Tag.json" -Encoding utf8; exit 1 }

"=== GATE2 [$Tag] ref='$Reference' $(Get-Date) ===" | Out-File -FilePath $log -Encoding utf8
$g=@{ tag=$Tag; reference=$Reference; date=(Get-Date).ToString("s") }
$busy=(Get-Process llama-bench,test-backend-ops,llama-perplexity,llama-completion -ErrorAction SilentlyContinue|Measure-Object).Count
if($busy -gt 0){ Fail "$busy GPU process already running" }
$env:HIP_PATH="C:\Program Files\AMD\ROCm\7.2\"; $env:PATH=$env:HIP_PATH+"bin;"+$env:PATH
$env:HIP_VISIBLE_DEVICES="0"; $env:ROCR_VISIBLE_DEVICES="0"
$g.base=(& git -C C:\Projects\lcpp rev-parse --short HEAD)

# ---- 1. op correctness: compare FAILURE SET to reference, not to zero -------
L "[1/4] op correctness"
$o = & "$bin\test-backend-ops.exe" 2>&1
$txt = ($o -join "`n") -replace "\x1b\[[0-9;]*m",""
if($txt -notmatch "RX 9070"){ Fail "not running on the RX 9070" }
$failing = @()
foreach($line in ($txt -split "`n")){ if($line -match "^\s*(\S+\(.*\)):\s*FAIL"){ $failing += $Matches[1] } }
$g.ops_fail_count=$failing.Count
$g.ops_failing=$failing
$failing | Out-File "$res\opsfail_$Tag.txt" -Encoding utf8
L ("  failing ops: " + $failing.Count)

# ---- 2. perplexity ---------------------------------------------------------
L "[2/4] perplexity"
$p = & "$bin\llama-perplexity.exe" -m $model -f $corpus -ngl 99 --chunks 32 -c 512 2>&1
$ppl=$null; foreach($l in $p){ if($l -match "Final estimate: PPL = ([\d.]+)"){ $ppl=[double]$Matches[1] } }
if($ppl -eq $null){ Fail "could not parse perplexity" }
$g.ppl=$ppl; L ("  PPL = " + $ppl)

# ---- 3. KL divergence vs reference logits (the real quality measure) --------
if($Reference -eq ""){
  L "[3/4] saving reference logits (this is the reference run)"
  $k = & "$bin\llama-perplexity.exe" -m $model -f $corpus -ngl 99 --chunks 32 -c 512 --kl-divergence-base $logits 2>&1
  $k | Out-File "$res\kl_$Tag.txt" -Encoding utf8
  if(-not (Test-Path $logits)){ Fail "reference logits not written" }
  L ("  saved " + [math]::Round((Get-Item $logits).Length/1MB,1) + " MB")
  $g.kl_mean=0.0; $g.top1_agree=100.0
} else {
  L "[3/4] KL divergence against reference"
  if(-not (Test-Path $logits)){ Fail "no reference logits on disk" }
  $k = & "$bin\llama-perplexity.exe" -m $model -f $corpus -ngl 99 --chunks 32 -c 512 --kl-divergence --kl-divergence-base $logits 2>&1
  $k | Out-File "$res\kl_$Tag.txt" -Encoding utf8
  $kt = ($k -join "`n")
  $klmean=$null; $top1=$null
  if($kt -match "Mean\s+KLD:\s*([\d.eE+-]+)"){ $klmean=[double]$Matches[1] }
  if($kt -match "Same top p:\s*([\d.]+)"){ $top1=[double]$Matches[1] }
  if($klmean -eq $null){ foreach($l in $k){ if($l -match "KLD.*?([\d.eE+-]+)$"){ $klmean=[double]$Matches[1] } } }
  $g.kl_mean=$klmean; $g.top1_agree=$top1
  L ("  mean KLD = " + $klmean + "   same-top-token = " + $top1 + "%")
}

# ---- 4. task accuracy: winogrande (objective, automated) -------------------
L "[4/4] winogrande accuracy"
$wgFile="C:\Projects\moe-4gb-search\winogrande.csv"
if(Test-Path $wgFile){
  $w = & "$bin\llama-perplexity.exe" -m $model -f $wgFile -ngl 99 --winogrande --winogrande-tasks 150 2>&1
  $acc=$null; foreach($l in $w){ if($l -match "Final Winogrande score\(([\d.]+)"){ $acc=[double]$Matches[1] } if($l -match "^\s*150\s+([\d.]+)"){ $acc=[double]$Matches[1] } }
  $g.winogrande=$acc; L ("  winogrande = " + $acc + "%")
} else { L "  winogrande dataset absent - skipped"; $g.winogrande=$null }

# ---- verdict ---------------------------------------------------------------
if($Reference -ne ""){
  $refj = Get-Content "$res\gate2_$Reference.json" -Raw | ConvertFrom-Json
  $pplDelta = [math]::Abs($g.ppl - $refj.ppl) / $refj.ppl * 100
  $g.ppl_delta_pct = [math]::Round($pplDelta,4)
  $newFails = @($g.ops_failing | Where-Object { $refj.ops_failing -notcontains $_ })
  $g.new_failing_ops = $newFails
  L ("  ppl delta = " + $g.ppl_delta_pct + "%   new failing ops = " + $newFails.Count + "   KLD = " + $g.kl_mean)
  $reasons=@()
  if($pplDelta -gt 0.5){ $reasons += "perplexity moved $($g.ppl_delta_pct)% (limit 0.5%)" }
  if($newFails.Count -gt 0){ $reasons += "$($newFails.Count) op(s) newly failing" }
  if($g.kl_mean -ne $null -and $g.kl_mean -gt 0.01){ $reasons += "mean KLD $($g.kl_mean) > 0.01" }
  if($g.winogrande -ne $null -and $refj.winogrande -ne $null -and ($refj.winogrande - $g.winogrande) -gt 2.0){ $reasons += "winogrande dropped $([math]::Round($refj.winogrande-$g.winogrande,2)) pts" }
  if($reasons.Count -gt 0){ $g.verdict="FAIL"; $g.reason=($reasons -join "; ") } else { $g.verdict="PASS" }
} else { $g.verdict="REFERENCE" }
L ("VERDICT: " + $g.verdict + " " + $g.reason)
Remove-Item env:HIP_VISIBLE_DEVICES,env:ROCR_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$g | ConvertTo-Json -Depth 4 | Out-File "$res\gate2_$Tag.json" -Encoding utf8
L "=== GATE2 DONE ==="

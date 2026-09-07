# runner.ps1 -Prs "26301,18816,28398,27962,28178"
#   Walks the queue unattended. Per PR: fetch diff -> relevance/apply check ->
#   build -> variance-aware A/B on pp512+tg128 -> append verdict to queue.csv.
#   Never aborts the queue on a single failure.
param([Parameter(Mandatory=$true)][string]$Prs)
$repo="C:\Projects\lcpp"; $res="C:\Projects\moe-4gb-search\results"
$queue="$res\queue.csv"; $qlog="$res\runner.log"
function QL($m){ "$((Get-Date).ToString('HH:mm:ss'))  $m" | Out-File -Append -FilePath $qlog -Encoding utf8 }
if(-not (Test-Path $queue)){ "pr,status,detail,when" | Out-File $queue -Encoding utf8 }
"=== RUNNER started $(Get-Date) ===" | Out-File -Append -FilePath $qlog -Encoding utf8
function Rec($pr,$st,$dt){ "$pr,$st,`"$dt`",$((Get-Date).ToString('s'))" | Out-File -Append -FilePath $queue -Encoding utf8; QL ("#$pr -> $st : $dt") }

foreach($pr in ($Prs -split ',')){
  $pr=$pr.Trim()
  QL "##### PR #$pr #####"
  # clean slate
  & git -C $repo checkout -- . 2>&1 | Out-Null
  Get-Process llama-bench,test-backend-ops,llama-perplexity,llama-completion -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 3
  # fetch diff
  $diff="C:\Users\I3lue\pr$pr.diff"
  if(-not (Test-Path $diff)){
    & curl.exe -sL -o $diff "https://github.com/ggml-org/llama.cpp/pull/$pr.diff"
  }
  if(-not (Test-Path $diff) -or (Get-Item $diff).Length -lt 100){ Rec $pr "NO_DIFF" "could not fetch"; continue }
  # relevance: does it apply to the frozen base?
  $chk = & git -C $repo apply --check $diff 2>&1
  if($LASTEXITCODE -ne 0){
    $why = ($chk | Select-String "error:" | Select-Object -First 1)
    Rec $pr "NEEDS_PORT" ("does not apply: " + ($why -replace '"','')); continue
  }
  # run the validator
  & powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\I3lue\v4.ps1 -Pr $pr
  $vlog="$res\v4_$pr.log"
  if(-not (Test-Path $vlog)){ Rec $pr "NO_LOG" "validator produced nothing"; continue }
  $txt = Get-Content $vlog
  if(($txt -join ' ') -match "ABORT: (.+?)==="){ Rec $pr "ABORTED" $Matches[1]; continue }
  $imp = $txt | Select-String "IMPROVEMENT"
  $reg = $txt | Select-String "REGRESSION"
  if($imp){
    $d = ($imp | ForEach-Object { ($_ -replace '\s+',' ') -replace '.*FINAL |.*\d\d:\d\d:\d\d ','' }) -join ' ; '
    Rec $pr "IMPROVEMENT" ($d.Substring(0,[Math]::Min(300,$d.Length)))
  } elseif($reg){
    Rec $pr "REGRESSION" (($reg | Select-Object -First 2) -join ' ; ')
  } else {
    Rec $pr "WITHIN_NOISE" "no metric outside baseline range"
  }
  & git -C $repo checkout -- . 2>&1 | Out-Null
}
QL "=== RUNNER DONE ==="

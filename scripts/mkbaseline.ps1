$repo="C:\Projects\lcpp"; $res="C:\Projects\moe-4gb-search\results"
$log="$res\baseline.log"
function L($m){ "$((Get-Date).ToString('HH:mm:ss'))  $m" | Out-File -Append -FilePath $log -Encoding utf8 }
"=== build frozen-base binaries ONCE $(Get-Date) ===" | Out-File -FilePath $log -Encoding utf8
while((Get-Process llama-bench,test-backend-ops,llama-perplexity,llama-completion -ErrorAction SilentlyContinue|Measure-Object).Count -gt 0){ Start-Sleep -Seconds 10 }
& git -C $repo checkout -- . 2>&1 | Out-Null
$st = & git -C $repo status --porcelain
if($st){ L ("!!! tree not clean: " + ($st -join ' ')); exit 1 }
L ("tree clean at " + (& git -C $repo rev-parse --short HEAD))
$env:HIP_PATH="C:\Program Files\AMD\ROCm\7.2\"; $env:PATH=$env:HIP_PATH+"bin;"+$env:PATH
& cmd /c "call `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set `"PATH=C:\Program Files\AMD\ROCm\7.2\bin;C:\Program Files\CMake\bin;C:\msys64\mingw64\bin;%PATH%`" && cmake --build $repo\build-hip --target llama-bench llama-perplexity llama-completion test-backend-ops" 2>&1 | Out-File "$res\baseline_build.log"
$tail = Get-Content "$res\baseline_build.log" -Tail 3
if(($tail -join ' ') -match "build stopped|FAILED|permission denied"){ L "!!! BUILD FAILED"; $tail | ForEach-Object { L ("   " + $_) }; exit 1 }
$dst="$repo\bin-baseline"
Remove-Item $dst -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $dst | Out-Null
Copy-Item "$repo\build-hip\bin\*" $dst -Force
$h=(Get-FileHash "$dst\ggml-hip.dll" -Algorithm SHA256).Hash.Substring(0,16)
L ("baseline binaries saved to $dst  ggml-hip.dll sha=$h  files=" + (Get-ChildItem $dst | Measure-Object).Count)
$h | Out-File "$res\baseline_sha.txt" -Encoding ascii
L "=== BASELINE BUILT ==="

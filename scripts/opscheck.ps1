# opscheck.ps1 -- the ONE correct way to read a test-backend-ops result.
#
# Why this file exists: the original inline check in this project counted only lines
# matching "OP(...): FAIL" and never read the process exit code or the
# "Backend <name>: OK|FAIL" summary line. test-backend-ops does NOT mark a backend-level
# failure with ": FAIL" on each case -- it prints "Failing tests:" and then lists the
# failing case descriptors bare. So a broken build could report "0 failing ops".
# That mistake cleared the DPP-reduce patch twice before it was caught.
#
# Also worth knowing when reading results: init_tensor_uniform() seeds a per-thread
# std::default_random_engine from std::random_device{}(), so EVERY RUN USES DIFFERENT
# INPUT DATA. A case sitting close to its error tolerance will fail only on an unlucky
# draw. A single clean run is therefore not proof of correctness -- repeat before
# declaring a stack good, and treat an intermittent failure as a real narrow margin,
# not as noise to be ignored.
#
# Usage:   . "$PSScriptRoot\opscheck.ps1"
#          $r = Invoke-OpsCheck -Exe "...\test-backend-ops.exe"
#          $r = Invoke-OpsCheck -Exe $exe -Filter "ADD,FLASH_ATTN_EXT"   # much faster
#          if (-not $r.pass) { ... }

function Invoke-OpsCheck {
    param(
        [Parameter(Mandatory=$true)][string] $Exe,
        [string] $Filter = $null
    )
    if ($Filter) { $o = & $Exe test -o $Filter 2>&1 } else { $o = & $Exe 2>&1 }
    $code  = $LASTEXITCODE
    $text  = ($o -join "`n") -replace "\x1b\[[0-9;]*m",""
    $lines = $text -split "`n"

    $ok = 0; $ns = 0
    foreach ($ln in $lines) {
        if     ($ln -match "^\s*\S+\(.*?\):\s*OK")            { $ok++ }
        elseif ($ln -match "^\s*\S+\(.*?\):\s*NOT SUPPORTED") { $ns++ }
    }

    # authoritative verdict line, e.g. "Backend ROCm0: FAIL"
    $verdict = ($lines | Where-Object { $_ -match '^Backend \S+:\s*(OK|FAIL)' } | Select-Object -First 1)
    $verdict = ("" + $verdict).Trim()

    # exact failing cases, taken from the explicit marker rather than guessed at
    $cases = @(); $inblock = $false
    foreach ($ln in $lines) {
        if ($ln -match '^\s*Failing tests:') { $inblock = $true; continue }
        if ($inblock) {
            if ($ln -match '^\s*([A-Z_0-9]+\(.*\))\s*$') { $cases += $Matches[1] }
            elseif ($ln.Trim() -ne '')                   { $inblock = $false }
        }
    }

    # numeric margin per failure: "[OP] ERR = <actual> > <threshold>"
    $errs = @()
    foreach ($ln in $lines) {
        if ($ln -match '\[([A-Z_0-9]+)\]\s*ERR\s*=\s*([0-9.eE+-]+)\s*>\s*([0-9.eE+-]+)') {
            $errs += [pscustomobject]@{ op=$Matches[1]; err=[double]$Matches[2]; threshold=[double]$Matches[3] }
        }
    }

    $families = (($cases | ForEach-Object { ($_ -split '\(')[0] } | Sort-Object -Unique) -join ',')

    # a run passes only if ALL THREE agree
    $pass = ($code -eq 0) -and ($verdict -notmatch 'FAIL') -and ($cases.Count -eq 0)

    return [pscustomobject]@{
        pass = $pass; code = $code; verdict = $verdict
        ok = $ok; not_supported = $ns
        families = $families; cases = $cases; errors = $errs
        summary = ("exit {0}  {1}  OK {2}  NS {3}  failing: {4}" -f $code, $verdict, $ok, $ns, $(if($families){$families}else{'none'}))
    }
}

# Repeat a check N times; returns the pass rate. Use this, not a single run.
function Invoke-OpsCheckRepeated {
    param(
        [Parameter(Mandatory=$true)][string] $Exe,
        [string] $Filter = $null,
        [int] $Runs = 5
    )
    $results = @()
    foreach ($i in 1..$Runs) { $results += Invoke-OpsCheck -Exe $Exe -Filter $Filter }
    $nfail = ($results | Where-Object { -not $_.pass } | Measure-Object).Count
    return [pscustomobject]@{
        runs = $Runs; failed = $nfail; pass = ($nfail -eq 0)
        results = $results
        summary = ("{0} of {1} runs failed" -f $nfail, $Runs)
    }
}

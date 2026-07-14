$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if ($env:CONDA_PREFIX) {
    $python = Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    $python = "python"
}
$sampleCounts = @(1293, 2048, 3072, 4096)
$log = Join-Path (Resolve-Path ".") "results\combo_sample_sweep_stdout.log"
$err = Join-Path (Resolve-Path ".") "results\combo_sample_sweep_stderr.log"

"started $(Get-Date -Format o)" | Set-Content -LiteralPath $log -Encoding utf8
"started $(Get-Date -Format o)" | Set-Content -LiteralPath $err -Encoding utf8

foreach ($n in $sampleCounts) {
    "=== rqmc_n_samples=$n ===" | Tee-Object -FilePath $log -Append
    & $python experiments\run_convex_combo.py --n-samples $n 1>> $log 2>> $err
    "exit_code=$LASTEXITCODE" | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "run_convex_combo.py failed for n=$n"
    }
}

"finished $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append

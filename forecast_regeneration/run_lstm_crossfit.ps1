<#
.SYNOPSIS
  CADR LSTM cross-fitting: static checks -> 4 cross-fit pipelines -> combined
  DDQN file -> audit -> hashes -> final gate.

.EXAMPLE
  .\run_lstm_crossfit.ps1 -Python "C:\path\to\python.exe" -Config ".\config_seed_0.yaml"
  .\run_lstm_crossfit.ps1 -Python "C:\path\to\python.exe" -Config ".\config_seed_0.yaml" -CheckOnly
#>
param(
    [string]$Python = "python",
    [string]$Config = ".\config_seed_0.yaml",
    [switch]$CheckOnly,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "[1/3] Static cross-fit checks (no training)..."
& $Python ".\crossfit_lstm.py" --config $Config --check-only
if ($LASTEXITCODE -ne 0) { Write-Host "Static checks FAILED." -ForegroundColor Red; exit $LASTEXITCODE }

if ($CheckOnly) {
    Write-Host "Static checks passed. No training was run." -ForegroundColor Green
    exit 0
}

Write-Host "[2/3] Cross-fitting the four targets (5 folds each) + determinism check..."
$crossfitArgs = @(".\crossfit_lstm.py", "--config", $Config)
if ($Overwrite) { $crossfitArgs += "--overwrite" }
& $Python @crossfitArgs
if ($LASTEXITCODE -ne 0) { Write-Host "Cross-fit FAILED." -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host "[3/3] Independent verification and final gate..."
& $Python ".\verify_crossfit.py" --config $Config
$code = $LASTEXITCODE
if ($code -eq 0) {
    Write-Host "Gate READY. Outputs: crossfit_results\" -ForegroundColor Green
} else {
    Write-Host "Gate BLOCKED. See crossfit_results\audit\crossfit_gate.json" -ForegroundColor Red
}
exit $code

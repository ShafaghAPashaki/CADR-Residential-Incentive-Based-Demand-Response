param(
    [Parameter(Mandatory=$true)]
    [string]$Python,

    [string]$Config = ".\config_seed_0.yaml",

    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Static validation"
& $Python ".\predict_load_lstm.py" --config $Config --check-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python ".\predict_price_lstm.py" --config $Config --check-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($CheckOnly) {
    Write-Host "Checks passed. No training was run."
    exit 0
}

# Fail early if forecast outputs are open in Excel/PDF/image viewers.
& $Python ".\build_forecast_outputs.py" --config $Config --check-output-access
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Full training must use the same US-holiday backend as the CADR environment.
& $Python -c "import holidays" 
if ($LASTEXITCODE -ne 0) {
    throw "Missing required Python package: holidays. Install requirements.txt and rerun."
}

Write-Host "[2/4] Train household-load LSTMs"
& $Python ".\predict_load_lstm.py" --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/4] Train wholesale-price LSTM"
& $Python ".\predict_price_lstm.py" --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/4] Build forecast outputs"
& $Python ".\build_forecast_outputs.py" --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done. Outputs: results\forecast\"

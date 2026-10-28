# Reproduce CADR Seed 0, July 2018 with authorised, locally supplied inputs.
# Does not alter frozen checkpoints or scientific parameters. No network access.
param(
    [string] $Python = ".\.venv\Scripts\python.exe",
    [string] $RawLoad = "downloads\15minute_data_austin.csv",
    [string] $RawPrice = "downloads\20180101-20181231 ERCOT Real-time Price.csv",
    [string] $BoundaryPrice = "",
    [switch] $ReusePreparedData,
    [switch] $ReuseForecasts,
    [switch] $ForecastOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$root = $PSScriptRoot
Set-Location $root

if (-not (Test-Path $Python)) { throw "Python not found: $Python" }
$py = (Resolve-Path $Python).Path
function RunPython([string[]]$Arguments) {
    Write-Host "python $($Arguments -join ' ')" -ForegroundColor Cyan
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')" }
}

# Guard against mixing a new run into existing outputs.
$selected = Join-Path $root "repro_runs\july_seed0\seed_0\training"
if ((-not $ForecastOnly) -and (Test-Path $selected)) { throw "Training results already exist at $selected. Preserve them; use a new checkout to rerun." }

if (-not $ReusePreparedData) {
    if (-not (Test-Path $RawLoad)) { throw "Missing licensed Dataport export: $RawLoad" }
    if (-not (Test-Path $RawPrice)) { throw "Missing EnergyOnline export: $RawPrice" }
    if ((Test-Path "data\load_hourly_2018.csv") -or (Test-Path "data\ercot_hourly_price.csv")) {
        throw "Prepared inputs already exist. Do not overwrite. Use -ReusePreparedData after verifying their origin or use a fresh checkout."
    }
    RunPython -Arguments @("data/prepare_load.py","--input",$RawLoad,"--output","data/load_hourly_2018.csv","--year","2018")
    $priceArgs = @("data/prepare_price.py","--input",$RawPrice,"--zone","LZ_AEN","--output","data/ercot_hourly_price.csv","--year","2018")
    if ($BoundaryPrice) {
        if (-not (Test-Path $BoundaryPrice)) { throw "Missing boundary export: $BoundaryPrice" }
        $priceArgs += @("--boundary-file",$BoundaryPrice)
    }
    RunPython -Arguments $priceArgs
} else {
    if (-not ((Test-Path "data\load_hourly_2018.csv") -and (Test-Path "data\ercot_hourly_price.csv"))) {
        throw "-ReusePreparedData requires both pre-existing, authorised processed inputs"
    }
}

RunPython -Arguments @("-m","unittest","discover","-s","tests","-p","test_data_preparation.py")
RunPython -Arguments @("-m","unittest","discover","-s","tests","-p","test_price_pipeline.py")

if (-not $ReuseForecasts) {
    $forecastDir = Join-Path $root "forecast_regeneration"
    if ((Test-Path "$forecastDir\crossfit_results") -or (Test-Path "$forecastDir\results")) {
        throw "Forecast regeneration outputs already exist. Use a fresh checkout or -ReuseForecasts after checking provenance."
    }
    New-Item -ItemType Directory "$forecastDir\data" -Force | Out-Null
    Copy-Item "data\load_hourly_2018.csv" "$forecastDir\data\load_hourly_2018.csv"
    Copy-Item "data\ercot_hourly_price.csv" "$forecastDir\data\ercot_hourly_price.csv"
    $savedPath = $env:PYTHONPATH
    $savedHashSeed = $env:PYTHONHASHSEED
    try {
        $env:PYTHONPATH = $root
        $env:PYTHONHASHSEED = "0"
        Push-Location $forecastDir
        try {
            RunPython -Arguments @("predict_load_lstm.py","--config","config_seed_0.yaml")
            RunPython -Arguments @("predict_price_lstm.py","--config","config_seed_0.yaml")
            RunPython -Arguments @("crossfit_lstm.py","--config","config_seed_0.yaml")
            RunPython -Arguments @("verify_crossfit.py","--config","config_seed_0.yaml")
            $gate = "crossfit_results\audit\crossfit_gate.txt"
            if (-not (Test-Path $gate) -or (Get-Content $gate -Raw).Trim() -ne "READY") {
                throw "Cross-fit verification gate is not READY."
            }
            RunPython -Arguments @("install_outputs.py")
        } finally { Pop-Location }
    } finally {
        $env:PYTHONPATH = $savedPath
        $env:PYTHONHASHSEED = $savedHashSeed
    }
}

if (-not (Test-Path "results\forecast\ddqn_state_forecasts.csv")) {
    throw "Forecast-state file missing; no DDQN training permitted."
}
if ($ForecastOnly) {
    Write-Host "FORECAST-ONLY COMPLETE: now run .\\run_checkpoint_july.ps1 to evaluate the published checkpoint." -ForegroundColor Green
    return
}
RunPython -Arguments @("main.py","--config_path","config_seed_0.yaml",
    "--allow_regenerated_forecasts","--results_root","repro_runs/july_seed0")

$model = "repro_runs/july_seed0/seed_0/training/ddqn_best.pth"
$config = "repro_runs/july_seed0/seed_0/training/config.yaml"
if (-not ((Test-Path $model) -and (Test-Path $config))) {
    throw "Selected DDQN checkpoint or its configuration is missing."
}
RunPython -Arguments @("test.py","--model_path",$model,"--config_path",$config,
    "--output_dir","repro_runs/july_seed0/evaluation_july",
    "--test_start","182","--test_end","212","--target_plot_days","208")

Write-Host "COMPLETE: independent July Seed-0 outputs in repro_runs/july_seed0" -ForegroundColor Green
Write-Host "Check checkpoint metadata, input hashes, and July per-seed metrics before claiming replication."

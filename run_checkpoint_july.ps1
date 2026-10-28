# Tier 2: evaluate released Seed-0 CADR checkpoint; never train.
param([string]$Python = ".\.venv\Scripts\python.exe", [string]$OutputDir = "repro_runs\checkpoint_seed0_july")
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot
if (-not (Test-Path $Python)) { throw "Missing Python executable: $Python" }
foreach ($path in @("results\forecast\ddqn_state_forecasts.csv","data\load_hourly_2018.csv","data\ercot_hourly_price.csv","results\seed_0\training\ddqn_best.pth")) {
    if (-not (Test-Path $path)) { throw "Missing input: $path. See README.md for preparing inputs and forecasts." }
}
if (Test-Path $OutputDir) { throw "Output directory already exists: $OutputDir" }
$argsList = @("tools/evaluate_with_regenerated_forecasts.py", "test.py", "--model_path", "results/seed_0/training/ddqn_best.pth", "--config_path", "config_seed_0.yaml", "--output_dir", $OutputDir, "--test_start", "182", "--test_end", "212", "--target_plot_days", "208")
& (Resolve-Path $Python).Path @argsList
if ($LASTEXITCODE -ne 0) { throw "Checkpoint evaluation failed with exit code $LASTEXITCODE" }
Write-Host "Checkpoint-only evaluation completed at $OutputDir. Inspect APPROXIMATE_FORECAST_INPUTS.json if present."

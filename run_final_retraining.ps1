# Offline/local, fail-closed 7-kW production run. Do not delete legacy results.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Verifying real Python dependencies and frozen forecast provenance..."
python -c "from utils.final_run_protocol import require_real_holidays; print(require_real_holidays())"
if ($LASTEXITCODE -ne 0) { throw "Runtime dependency preflight failed" }

Write-Host "Verifying retained lambda_c and its immutable audit across all three CADR configs..."
python verify_retained_lambda.py
if ($LASTEXITCODE -ne 0) { throw "Retained lambda_c preflight failed" }
python refresh_source_manifests.py
if ($LASTEXITCODE -ne 0) { throw "Source manifest refresh failed" }

# Frozen, descriptive confusion counts; NEVER used to choose a model.
python evaluate_forecast_confusion.py
if ($LASTEXITCODE -ne 0) { throw "Forecast confusion report failed" }

foreach ($seed in 0,1,2) {
  Write-Host "Starting final CADR training: seed $seed, 2500 episodes"
  python main.py --config_path "config_seed_$seed.yaml" --seed $seed
  if ($LASTEXITCODE -ne 0) { throw "CADR seed $seed failed" }
}
foreach ($seed in 0,1,2) {
  Write-Host "Starting final EBDR training: seed $seed, 2500 episodes"
  python -m benchmark.main_benchmark --config "benchmark/config_benchmark_seed_$seed.yaml"
  if ($LASTEXITCODE -ne 0) { throw "EBDR seed $seed failed" }
}
Write-Host "All six final training runs completed; checkpoints are validation-selected."
Write-Host "Next: run frozen evaluation scripts and inference-only oracle comparisons."

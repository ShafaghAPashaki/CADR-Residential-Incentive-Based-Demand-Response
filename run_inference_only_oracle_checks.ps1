# Inference-only value-of-information; run AFTER 2500-episode final checkpoint runs.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
foreach ($method in @('cadr','ebdr')) {
  foreach ($seed in 0,1,2) {
    foreach ($period in @('july','august','november')) {
      Write-Host "Inference-only: $method seed=$seed period=$period"
      python evaluate_final_diagnostics.py --method $method --seed $seed --period $period
      if ($LASTEXITCODE -ne 0) { throw "Inference-only evaluation failed" }
    }
  }
}
Write-Host "Oracle sensitivity comparisons complete; no policies were retrained."

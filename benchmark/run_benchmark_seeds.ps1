param(
    [string]$Python = "python",
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    foreach ($Seed in 0..2) {
        $Config = "benchmark/config_benchmark_seed_$Seed.yaml"
        Write-Host "============================================================"
        Write-Host "Training benchmark seed $Seed"
        Write-Host "Config: $Config"
        Write-Host "============================================================"
        $Arguments = @("-m", "benchmark.main_benchmark", "--config", $Config)
        if ($SmokeTest) { $Arguments += "--smoke-test" }
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Benchmark training failed for seed $Seed."
        }
    }
}
finally {
    Pop-Location
}

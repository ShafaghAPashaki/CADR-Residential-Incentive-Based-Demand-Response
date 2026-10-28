param(
    [string]$Python = "python",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    foreach ($Seed in 0..2) {
        $Config = "benchmark/config_benchmark_seed_$Seed.yaml"
        $Model = "results/benchmark/seed_$Seed/training/ddqn_best.pth"
        if (-not (Test-Path $Model)) {
            throw "Missing benchmark checkpoint: $Model"
        }
        foreach ($Period in @("july", "august", "november")) {
            Write-Host "Testing benchmark seed $Seed | $Period"
            $Arguments = @(
                "-m", "benchmark.test_benchmark",
                "--model_path", $Model,
                "--config_path", $Config,
                "--period", $Period,
                "--output_dir", "repro_runs/ebdr/seed_$Seed/$Period"
            )
            if ($Overwrite) { $Arguments += "--overwrite" }
            & $Python @Arguments
            if ($LASTEXITCODE -ne 0) {
                throw "Benchmark testing failed for seed $Seed, period $Period."
            }
        }
    }

    & $Python -m benchmark.summarize_benchmark --results_root repro_runs/ebdr --output repro_runs/ebdr/summary.xlsx
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark summarisation failed."
    }

    if (Test-Path "results/summary.xlsx") {
        & $Python -m benchmark.compare_methods --proposed_summary results/summary.xlsx --benchmark_summary repro_runs/ebdr/summary.xlsx --output repro_runs/method_comparison.xlsx
        if ($LASTEXITCODE -ne 0) {
            throw "Cross-method comparison failed."
        }
    }
    else {
        Write-Warning "results/summary.xlsx was not found; repro_runs/method_comparison.xlsx was not generated."
    }
}
finally {
    Pop-Location
}

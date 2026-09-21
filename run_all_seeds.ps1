param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
& $Python ".\verify.py"
if ($LASTEXITCODE -ne 0) { throw "Verification failed." }

foreach ($seed in 0,1,2) {
    & $Python ".\main.py" --config_path ".\config_seed_$seed.yaml"
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $seed." }
}

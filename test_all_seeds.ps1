param(
    [string]$Python = "python",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$testExtraArgs = @()
if ($Overwrite) {
    $testExtraArgs += "--overwrite"
}
else {
    $existingDerivedOutputs = @(
        ".\results\q_states.csv",
        ".\results\summary.xlsx",
        ".\results\seed_0\q_diagnostic.xlsx",
        ".\results\seed_1\q_diagnostic.xlsx",
        ".\results\seed_2\q_diagnostic.xlsx"
    )
    foreach ($path in $existingDerivedOutputs) {
        if (Test-Path $path) {
            throw "Evaluation output already exists: $path. Use -Overwrite for an intentional rerun."
        }
    }
}

foreach ($seed in 0,1,2) {
    $config = ".\config_seed_$seed.yaml"
    $model = ".\results\seed_$seed\training\ddqn_best.pth"

    & $Python ".\test.py" --model_path $model --config_path $config --output_dir ".\results\seed_$seed\july" --test_start 182 --test_end 212 --target_plot_days 208 @testExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "July test failed for seed $seed." }

    & $Python ".\test.py" --model_path $model --config_path $config --output_dir ".\results\seed_$seed\august" --test_start 213 --test_end 243 --target_plot_days 220 @testExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "August test failed for seed $seed." }

    & $Python ".\test.py" --model_path $model --config_path $config --output_dir ".\results\seed_$seed\november" --test_start 305 --test_end 334 --target_plot_days 312 @testExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "November test failed for seed $seed." }
}

$manifest = ".\results\q_states.csv"
& $Python ".\q_action_diagnostic.py" --model_path ".\results\seed_0\training\ddqn_best.pth" --config_path ".\config_seed_0.yaml" --label "seed_0" --output ".\results\seed_0\q_diagnostic.xlsx" --state_manifest $manifest --samples_per_stratum 2 --policy_failure_samples 2 --seed 20260727 --training_metrics ".\results\seed_0\training\training_metrics.npy" --recreate_manifest
if ($LASTEXITCODE -ne 0) { throw "Q diagnostic failed for seed 0." }

foreach ($seed in 1,2) {
    & $Python ".\q_action_diagnostic.py" --model_path ".\results\seed_$seed\training\ddqn_best.pth" --config_path ".\config_seed_$seed.yaml" --label "seed_$seed" --output ".\results\seed_$seed\q_diagnostic.xlsx" --state_manifest $manifest --samples_per_stratum 2 --policy_failure_samples 2 --seed 20260727 --training_metrics ".\results\seed_$seed\training\training_metrics.npy"
    if ($LASTEXITCODE -ne 0) { throw "Q diagnostic failed for seed $seed." }
}

& $Python ".\summarize_results.py"
if ($LASTEXITCODE -ne 0) { throw "Result summarisation failed." }

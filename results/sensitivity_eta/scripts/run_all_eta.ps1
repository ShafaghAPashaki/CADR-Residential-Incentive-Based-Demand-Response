param(
    [string]$Tag = (Get-Date -Format "yyyyMMdd_HHmmss"),
    [Nullable[int]]$Seed = $null
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ArgsList = @("$ScriptDir\run_all_eta.py", "--tag", $Tag)
if ($null -ne $Seed) {
    $ArgsList += @("--seed", "$Seed")
}
python @ArgsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

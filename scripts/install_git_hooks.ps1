Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$hooksPath = Join-Path $repoRoot ".githooks"
$hookFile = Join-Path $hooksPath "pre-commit"

if (-not (Test-Path $hookFile)) {
    throw "Hook file not found: $hookFile"
}

git -C $repoRoot config core.hooksPath .githooks | Out-Null

Write-Host "Git hooks installed."
Write-Host "core.hooksPath = .githooks"
Write-Host "Active hook: .githooks/pre-commit"

<#
Easy launcher for the F1 tipping optimiser (PowerShell).

  .\run.ps1                       # setup venv (if needed) + `main.py all --source all`
  .\run.ps1 fit                   # any main.py subcommand/flags are passed straight through
  .\run.ps1 all --manual odds.csv
  .\run.ps1 --setup               # just (re)create the venv + install deps, don't run

Idempotent: the venv is created and deps installed only when missing. Delete
venv\.deps-installed to force a reinstall.
#>
$ErrorActionPreference = 'Stop'
$Root  = $PSScriptRoot
$Venv  = Join-Path $Root 'venv'
$Stamp = Join-Path $Venv '.deps-installed'
$Py    = Join-Path $Venv 'Scripts\python.exe'

if (-not (Test-Path $Py)) {
  Write-Host ">> creating venv at $Venv"
  $boot = (Get-Command python -ErrorAction SilentlyContinue) ?? (Get-Command py -ErrorAction SilentlyContinue)
  if (-not $boot) { throw "no python interpreter found on PATH" }
  & $boot.Source -m venv $Venv
}

if (-not (Test-Path $Stamp)) {
  Write-Host ">> installing dependencies"
  & $Py -m pip install --upgrade pip | Out-Null
  & $Py -m pip install -r (Join-Path $Root 'config\requirements.txt')
  & $Py -m pip install -r (Join-Path $Root 'optimiser\requirements.txt')
  New-Item -ItemType File -Path $Stamp | Out-Null
}

if ($args.Count -ge 1 -and $args[0] -eq '--setup') {
  Write-Host ">> setup complete"
  exit 0
}

$passthru = if ($args.Count -eq 0) { @('all', '--source', 'all') } else { $args }

# _paths.py bootstraps ..\src, so main.py must run with optimiser\ as the cwd.
Set-Location (Join-Path $Root 'optimiser')
& $Py main.py @passthru
exit $LASTEXITCODE

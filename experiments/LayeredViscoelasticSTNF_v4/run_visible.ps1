$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
chcp 65001 | Out-Null
$Host.UI.RawUI.WindowTitle = 'V4-A Prony Live Monitor'
$Experiment = 'D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\LayeredViscoelasticSTNF_v4'
$Output = Join-Path $Experiment 'artifacts'
$Python = 'D:\anaconda3\python.exe'
$Log = Join-Path $Output 'training.log'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
Set-Location $Experiment
Write-Host 'V4-A development-only experiment started. The fixed test set will not be read.' -ForegroundColor Cyan
Write-Host "Artifact directory: $Output" -ForegroundColor Cyan
& $Python '.\run_v4a.py' --data-root 'D:\matlab\guochuang' --output $Output 2>&1 | Tee-Object -FilePath $Log
$Code = $LASTEXITCODE
if ($Code -eq 0) {
    Write-Host 'Experiment completed. This window remains open for review.' -ForegroundColor Green
} else {
    Set-Content -LiteralPath (Join-Path $Output 'FAILED') -Value "exit_code=$Code`ntime=$(Get-Date -Format o)" -Encoding utf8
    Write-Host "Experiment failed with exit code $Code. See $Log" -ForegroundColor Red
}

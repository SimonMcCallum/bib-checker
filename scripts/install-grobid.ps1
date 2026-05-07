#requires -version 5.1
$ErrorActionPreference = "Stop"
$image = "lfoppiano/grobid:0.8.1"
Write-Output "Pulling $image (~500MB on first run)..."
& docker pull $image
if ($LASTEXITCODE -ne 0) { throw "docker pull failed" }
Write-Output "Done. Run scripts\start-grobid.ps1 to start the service."

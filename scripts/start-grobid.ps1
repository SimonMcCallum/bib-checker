#requires -version 5.1
$ErrorActionPreference = "Stop"
$image = "lfoppiano/grobid:0.8.1"
$name = "grobid"

$existing = & docker ps -a --filter "name=^/$name$" --format "{{.Names}}"
if ($existing -eq $name) {
    Write-Output "Container '$name' exists; removing..."
    & docker rm -f $name | Out-Null
}

Write-Output "Starting $image on http://localhost:8070 ..."
& docker run --rm -d -p 8070:8070 --name $name $image | Out-Null
if ($LASTEXITCODE -ne 0) { throw "docker run failed" }

# Wait for /api/isalive
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8070/api/isalive" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) {
            Write-Output "GROBID is alive."
            exit 0
        }
    } catch { Start-Sleep -Seconds 2 }
}
throw "GROBID did not become healthy within 120s"

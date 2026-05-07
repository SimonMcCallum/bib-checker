#requires -version 5.1
& docker stop grobid 2>$null | Out-Null
Write-Output "Stopped."

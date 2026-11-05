$ErrorActionPreference = 'Stop'
$demoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$dockerExe = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $dockerExe) { $dockerExe = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe' }
Push-Location $demoRoot
try {
  & $dockerExe compose -f compose.deploy.yml down
  if ($LASTEXITCODE -ne 0) { throw 'Docker Compose could not stop the deployment.' }
  Write-Output 'Deployment stopped. PostgreSQL and repository volumes were preserved.'
} finally {
  Pop-Location
}

param([switch]$Gpu, [switch]$Offline)
$ErrorActionPreference = 'Stop'
$demoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$dockerExe = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $dockerExe) {
  $dockerExe = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
}
if (-not (Test-Path -LiteralPath $dockerExe)) { throw 'Docker CLI not found. Install Docker Desktop first.' }
& $dockerExe info --format '{{.ServerVersion}}' *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop Linux engine is not running. Start Docker Desktop first.' }
Push-Location $demoRoot
try {
  $composeFiles = @('-f', 'compose.deploy.yml')
  if ($Gpu) { $composeFiles += @('-f', 'compose.gpu.yml') }
  if ($Offline) {
    if (-not (Test-Path -LiteralPath (Join-Path $demoRoot 'frontend/dist/index.html') -PathType Leaf)) {
      throw 'Frontend build is missing. Run npm ci and npm run build in the frontend directory.'
    }
    $requiredImages = @('postgres:16', 'redis:7-alpine', 'repocoding-api:local', 'repocoding-java-runner:local', 'repocoding-web:local')
    if ($Gpu) { $requiredImages += 'repocoding-research-gpu:local' }
    else { $requiredImages += 'repocoding-research:local' }
    foreach ($imageName in $requiredImages) {
      & $dockerExe image inspect $imageName --format '{{.Id}}' *> $null
      if ($LASTEXITCODE -ne 0) { throw "Local image missing: $imageName. Build it when Docker Hub is reachable." }
    }
    $composeFiles += @('-f', 'compose.offline.yml')
    & $dockerExe compose @composeFiles config --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Offline Compose configuration is invalid.' }
    & $dockerExe compose @composeFiles up -d --no-build --pull never
  } else {
    & $dockerExe compose @composeFiles up -d --build
  }
  if ($LASTEXITCODE -ne 0) { throw "Docker Compose deployment failed. Run docker compose $($composeFiles -join ' ') ps and logs for details." }
  $port = if ($env:WEB_PORT) { $env:WEB_PORT } else { '8080' }
  $url = "http://127.0.0.1:$port/health"
  $ready = $false
  for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
      $response = Invoke-RestMethod -Uri $url -TimeoutSec 3
      if ($response.status -eq 'ok') { $ready = $true; break }
    } catch { Start-Sleep -Seconds 2 }
  }
  if (-not $ready) { throw "Deployment did not become ready: $url" }
  Write-Output "Ready: http://127.0.0.1:$port"
} finally {
  Pop-Location
}

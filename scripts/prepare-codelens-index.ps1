param(
  [string]$RepoKey = 'apache__dubbo',
  [string]$RepographPath = '3.1-d0a1bd0'
)
$ErrorActionPreference = 'Stop'
if ($RepoKey -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]*$') {
  throw 'RepoKey must be an identifier, not a path.'
}
if ($RepographPath -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]*$') {
  throw 'RepographPath must be a single directory name.'
}
$demoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$graphRoot = (Resolve-Path (Join-Path $demoRoot '..\dubbo')).Path
$graphPath = Join-Path $graphRoot $RepographPath
foreach ($name in @('graph.pkl', 'tags.json')) {
  if (-not (Test-Path -LiteralPath (Join-Path $graphPath $name) -PathType Leaf)) {
    throw "Missing Repograph input: $name in $graphPath"
  }
}
$dockerExe = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $dockerExe) { throw 'Docker CLI not found.' }
$composeFiles = @('-f', 'compose.deploy.yml', '-f', 'compose.gpu.yml')
Push-Location $demoRoot
try {
  & $dockerExe info --format '{{.ServerVersion}}' *> $null
  if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop Linux engine is unavailable.' }
  & $dockerExe compose @composeFiles config --quiet
  if ($LASTEXITCODE -ne 0) { throw 'Compose GPU configuration is invalid.' }
  & $dockerExe image inspect repocoding-research-gpu:local --format '{{.Id}}' *> $null
  if ($LASTEXITCODE -ne 0) {
    & $dockerExe compose @composeFiles build codelens-worker
    if ($LASTEXITCODE -ne 0) { throw 'GPU research image build failed.' }
  }
  & $dockerExe compose @composeFiles run --rm --no-deps --entrypoint python codelens-worker -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA ready:", torch.cuda.get_device_name(0))'
  if ($LASTEXITCODE -ne 0) { throw 'CUDA is unavailable inside the research image.' }
  & $dockerExe compose @composeFiles run --rm --no-deps --entrypoint python codelens-worker /app/prepare_codelens_index.py --repo-key $RepoKey --repograph-path $RepographPath
  if ($LASTEXITCODE -ne 0) { throw 'CodeLens index preparation failed.' }
  Write-Output 'CodeLens index preparation completed.'
} finally {
  Pop-Location
}

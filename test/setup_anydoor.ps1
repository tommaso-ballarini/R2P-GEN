# test/setup_anydoor.ps1
# PowerShell script to setup AnyDoor for testing
# Run from project root: .\test\setup_anydoor.ps1

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Setting up AnyDoor for Testing" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ExternalDir = Join-Path $ScriptDir "external"
$CheckpointDir = Join-Path $ProjectRoot "checkpoints" "anydoor"

# Create directories
Write-Host "[1/4] Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $ExternalDir | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null
Write-Host "      Created: $ExternalDir" -ForegroundColor Green
Write-Host "      Created: $CheckpointDir" -ForegroundColor Green

# Clone AnyDoor repository
Write-Host ""
Write-Host "[2/4] Cloning AnyDoor repository..." -ForegroundColor Yellow
$AnyDoorPath = Join-Path $ExternalDir "AnyDoor"

if (Test-Path $AnyDoorPath) {
    Write-Host "      AnyDoor already cloned at $AnyDoorPath" -ForegroundColor Green
} else {
    Set-Location $ExternalDir
    git clone https://github.com/ali-vilab/AnyDoor.git
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      Cloned successfully!" -ForegroundColor Green
    } else {
        Write-Host "      ERROR: Failed to clone AnyDoor repository" -ForegroundColor Red
        exit 1
    }
    Set-Location $ProjectRoot
}

# Install AnyDoor dependencies
Write-Host ""
Write-Host "[3/4] Installing AnyDoor dependencies..." -ForegroundColor Yellow
$AnyDoorReqs = Join-Path $AnyDoorPath "requirements.txt"

if (Test-Path $AnyDoorReqs) {
    pip install -r $AnyDoorReqs
    Write-Host "      Dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "      WARNING: requirements.txt not found, skipping" -ForegroundColor Yellow
}

# Download checkpoint
Write-Host ""
Write-Host "[4/4] Downloading AnyDoor checkpoint..." -ForegroundColor Yellow
$CheckpointFile = Join-Path $CheckpointDir "anydoor_model.pth"

if (Test-Path $CheckpointFile) {
    Write-Host "      Checkpoint already exists at $CheckpointFile" -ForegroundColor Green
} else {
    Write-Host "      Downloading from HuggingFace (this may take a while)..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "      NOTE: You need to manually download the checkpoint from:" -ForegroundColor Yellow
    Write-Host "      https://huggingface.co/xichenhku/AnyDoor" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "      Place the downloaded file at:" -ForegroundColor Yellow
    Write-Host "      $CheckpointFile" -ForegroundColor Cyan
    Write-Host ""
    
    # Try to download with huggingface-cli if available
    try {
        huggingface-cli download xichenhku/AnyDoor --local-dir $CheckpointDir
        Write-Host "      Download complete!" -ForegroundColor Green
    } catch {
        Write-Host "      Please download manually from HuggingFace" -ForegroundColor Yellow
    }
}

# Summary
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Setup Summary" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  AnyDoor repo:    $AnyDoorPath" -ForegroundColor White
Write-Host "  Checkpoint dir:  $CheckpointDir" -ForegroundColor White
Write-Host ""

# Check if everything is ready
$Ready = $true

if (-not (Test-Path $AnyDoorPath)) {
    Write-Host "  [ ] AnyDoor repository NOT cloned" -ForegroundColor Red
    $Ready = $false
} else {
    Write-Host "  [x] AnyDoor repository cloned" -ForegroundColor Green
}

if (-not (Test-Path $CheckpointFile)) {
    Write-Host "  [ ] Checkpoint NOT downloaded" -ForegroundColor Red
    $Ready = $false
} else {
    Write-Host "  [x] Checkpoint downloaded" -ForegroundColor Green
}

Write-Host ""
if ($Ready) {
    Write-Host "  Status: READY TO USE" -ForegroundColor Green
} else {
    Write-Host "  Status: SETUP INCOMPLETE" -ForegroundColor Yellow
    Write-Host "  Please complete the missing steps above" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan

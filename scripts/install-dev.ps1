# -----------------------------------------------------------------------------
# install-dev.ps1
# Setup Python 3.11 virtual environment and install development dependencies.
# ASCII only. Safe for Windows PowerShell.
# -----------------------------------------------------------------------------

Write-Host "=== KAN Dev Environment Setup ==="

# Step 1: Check Python version
Write-Host "Checking Python version..."
$pyVersion = python --version 2>$null
if (-not $pyVersion) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}
Write-Host "Detected: $pyVersion"

# Step 2: Create virtual environment
Write-Host "Creating virtual environment: .venv ..."
python -m venv .venv
if (-not (Test-Path ".venv")) {
    Write-Error "Failed to create virtual environment."
    exit 1
}

# Step 3: Activate venv
Write-Host "Activating virtual environment..."
$activate = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Error "Activate.ps1 not found."
    exit 1
}
. $activate

# Step 4: Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Step 5: Install runtime dependencies
if (-not (Test-Path "requirements.txt")) {
    Write-Error "requirements.txt not found."
    exit 1
}

Write-Host "Installing runtime dependencies..."
pip install -r requirements.txt

# Step 6: Optional dev dependencies (from pyproject.toml)
Write-Host "Installing dev dependencies..."
pip install -e ".[dev]"

Write-Host "=== Setup complete! ==="
Write-Host "To activate the environment later, run:"
Write-Host "    .\.venv\Scripts\Activate.ps1"

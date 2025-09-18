# -------------------------------
# Assert we are in capstone folder
# -------------------------------
$currentFolder = Split-Path -Leaf (Get-Location)
if ($currentFolder -ne "capstone") {
    Write-Error "You must be in the 'capstone' folder to run this script."
    exit 1
}

# -------------------------------
# Validate data folder structure
# -------------------------------
$dataPath = "$PWD\data"

if (-Not (Test-Path $dataPath)) {
    Write-Error "Data folder not found. Please ensure 'data' is available in this directory."
    exit 1
}

$years = @("2021", "2022")
foreach ($year in $years) {
    $yearPath = Join-Path $dataPath $year
    if (-Not (Test-Path $yearPath)) {
        Write-Error "$year folder not found in data. Please ensure it exists."
        exit 1
    }

    $subfolders = Get-ChildItem -Path $yearPath -Directory -ErrorAction SilentlyContinue
    if (-Not $subfolders) {
        Write-Error "$year folder exists but contains no subfolders. Please check your data."
        exit 1
    }
}

Write-Host "Data folder structure validated successfully."

# -------------------------------
# Ensure virtual environment exists
# -------------------------------
$venvPath = "$PWD\venv"
if (-Not (Test-Path $venvPath)) {
    Write-Host "Virtual environment not found. Creating..."

    try {
        python -m venv $venvPath
    } catch {
        Write-Host "venv creation failed. Trying virtualenv..."
        try {
            virtualenv $venvPath
        } catch {
            Write-Error "Both venv and virtualenv failed. Exiting."
            exit 1
        }
    }

    Write-Host "Virtual environment created."
} else {
    Write-Host "Virtual environment exists. Skipping creation."
}

# -------------------------------
# Activate venv and install requirements
# -------------------------------
$activateScript = "$venvPath\Scripts\Activate.ps1"
if (-Not (Test-Path $activateScript)) {
    Write-Error "Activation script not found. Exiting."
    exit 1
}

Write-Host "Activating virtual environment..."
& $activateScript

$requirementsPath = "$PWD\requirements.txt"
if (Test-Path $requirementsPath) {
    Write-Host "Installing packages from requirements.txt..."
    pip install -r $requirementsPath
} else {
    Write-Error "requirements.txt not found. Exiting."
    exit 1
}

# -------------------------------
# Run main.py
# -------------------------------
Write-Host "Running main.py..."
python main.py

# -------------------------------
# Keep shell open
# -------------------------------
Write-Host "`nScript finished. Press Enter to exit..."
Read-Host

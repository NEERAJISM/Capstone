# -------------------------------
# Assert we are in capstone folder
# -------------------------------
$currentFolder = Split-Path -Leaf (Get-Location)
if ($currentFolder -ne "capstone") {
    Write-Error "You must be in the 'capstone' folder to run this script."
    exit 1
}

# -------------------------------
# Ensure data folder exists
# -------------------------------
$dataPath = "$PWD\data"
if (-Not (Test-Path $dataPath)) {
    Write-Host "Data folder not found. Downloading and extracting data.zip..."

    $dropboxUrl = "https://www.dropbox.com/scl/fi/yfymt0eh0sp3xw92htdxv/data.zip?rlkey=xfqdzj6aack0eheq66bjpvm5q&st=02njn8rg&dl=1"
    $zipPath = "$PWD\data.zip"

    Invoke-WebRequest -Uri $dropboxUrl -OutFile $zipPath
    Expand-Archive -LiteralPath $zipPath -DestinationPath $PWD -Force
    Remove-Item $zipPath

    if (-Not (Test-Path $dataPath)) {
        Write-Error "Data folder was not created after extraction. Exiting."
        exit 1
    }

    Write-Host "Data setup complete."
} else {
    Write-Host "Data folder exists. Skipping download."
}

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

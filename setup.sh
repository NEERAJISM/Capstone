#!/bin/bash

# -------------------------------
# Ensure we are in capstone folder
# -------------------------------
currentFolder=${PWD##*/}
if [ "$currentFolder" != "capstone" ]; then
    echo "You must be in the 'capstone' folder to run this script."
    exit 1
fi

# -------------------------------
# Ensure data folder exists
# -------------------------------
dataPath="$PWD/data"
if [ ! -d "$dataPath" ]; then
    echo "Data folder not found. Downloading and extracting data.zip..."

    dropboxUrl="https://www.dropbox.com/scl/fi/yfymt0eh0sp3xw92htdxv/data.zip?rlkey=xfqdzj6aack0eheq66bjpvm5q&st=02njn8rg&dl=1"
    zipPath="$PWD/data.zip"

    # Download
    curl -L -o "$zipPath" "$dropboxUrl"

    # Extract
    unzip -o "$zipPath" -d "$PWD"
    rm "$zipPath"

    if [ ! -d "$dataPath" ]; then
        echo "Data folder was not created after extraction. Exiting."
        exit 1
    fi

    echo "Data setup complete."
else
    echo "Data folder exists. Skipping download."
fi

# -------------------------------
# Ensure virtual environment exists
# -------------------------------
venvPath="$PWD/venv"
if [ ! -d "$venvPath" ]; then
    echo "Virtual environment not found. Creating..."

    # Try python3 venv
    python3 -m venv "$venvPath" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "venv creation failed. Trying virtualenv..."
        virtualenv "$venvPath" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "Both venv and virtualenv failed. Exiting."
            exit 1
        fi
    fi

    echo "Virtual environment created."
else
    echo "Virtual environment exists. Skipping creation."
fi

# -------------------------------
# Activate venv and install requirements
# -------------------------------
source "$venvPath/bin/activate"

requirementsPath="$PWD/requirements.txt"
if [ -f "$requirementsPath" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r "$requirementsPath"
else
    echo "requirements.txt not found. Exiting."
    exit 1
fi

# -------------------------------
# Run main.py
# -------------------------------
echo "Running main.py..."
python main.py

# -------------------------------
# Keep terminal open
# -------------------------------
echo
read -p "Script finished. Press Enter to exit..."

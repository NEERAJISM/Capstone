#!/bin/bash

# -------------------------------
# Ensure we are in capstone folder
# -------------------------------
currentFolder=${PWD##*/}
if [ "$(echo "$currentFolder" | tr '[:upper:]' '[:lower:]')" != "capstone" ]; then
    echo "You must be in the 'capstone' folder to run this script."
    exit 1
fi

# -------------------------------
# Validate data folder structure
# -------------------------------
dataPath="$PWD/data"
if [ ! -d "$dataPath" ]; then
    echo "Data folder not found. Please ensure 'data' is available in this directory."
    exit 1
fi

for year in 2021 2022; do
    yearPath="$dataPath/$year"
    if [ ! -d "$yearPath" ]; then
        echo "$year folder not found in data. Please ensure it exists."
        exit 1
    fi

    subfolders=$(find "$yearPath" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    if [ -z "$subfolders" ]; then
        echo "$year folder exists but contains no subfolders. Please check your data."
        exit 1
    fi
done

echo "Data folder structure validated successfully."

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

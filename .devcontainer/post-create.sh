#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Install Python Requirements ---
# Check if requirements.txt exists before trying to install
if [ -f "requirements.txt" ]; then
    echo ">>> Installing Python requirements..."
    pip install -r requirements.txt
else
    echo ">>> Warning: requirements.txt not found. Skipping pip install."
fi

echo ">>> Updating package list..."
apt-get update

echo ">>> Installing system prerequisites (wget, unzip)..."
apt-get install -y wget unzip

# Ensure the target directory exists
echo ">>> Ensuring Datasets directory exists..."
mkdir -p Datasets

# --- Process TSB-AD-U ---
echo ">>> Downloading TSB-AD-U dataset..."
wget https://www.thedatum.org/datasets/TSB-AD-U.zip
echo ">>> Unzipping TSB-AD-U dataset..."
# Use -o to overwrite existing files without prompting if unzipping again
unzip -o TSB-AD-U.zip -d Datasets
echo ">>> Removing TSB-AD-U.zip..."
rm TSB-AD-U.zip # Remove the zip file after successful unzip

# --- Process TSB-AD-M ---
echo ">>> Downloading TSB-AD-M dataset..."
wget https://www.thedatum.org/datasets/TSB-AD-M.zip
echo ">>> Unzipping TSB-AD-M dataset..."
# Use -o to overwrite existing files without prompting
unzip -o TSB-AD-M.zip -d Datasets
echo ">>> Removing TSB-AD-M.zip..."
rm TSB-AD-M.zip # Remove the zip file after successful unzip

echo ">>> Setup script finished."
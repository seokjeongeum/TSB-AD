#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
git config --global user.email "jeseok@dblab.postech.ac.kr"
git config --global user.name "jeseok"
# This command runs as the 'vscode' user and modifies files in the workspace,
# so it does not need sudo.
git submodule update --init --recursive
# --recursive is for submodules that themselves have submodules

# --- Install Python Requirements ---
# Check if requirements.txt exists before trying to install
if [ -f "requirements.txt" ]; then
    echo ">>> Installing Python requirements for the user..."
    # Use '--user' to install packages into the user's home directory.
    # This is the recommended practice for non-root users and avoids permission issues.
    pip install --user -r requirements.txt
else
    echo ">>> Warning: requirements.txt not found. Skipping pip install."
fi

echo ">>> Updating package list..."
# apt-get requires root privileges, so we add 'sudo'.
# The 'vscode' user has passwordless sudo access thanks to the common-utils feature.
sudo apt-get update

echo ">>> Installing system prerequisites (wget, unzip)..."
# Installing system-wide packages also requires sudo.
sudo apt-get install -y wget unzip

# --- The following commands operate within the workspace, so they do not need sudo ---

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

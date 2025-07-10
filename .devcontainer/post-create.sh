#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Initial Git and Ownership Configuration ---
echo ">>> Setting up Git and file ownership..."
sudo chown -R $(whoami) .
git config --global --add safe.directory /workspaces/TSB-AD
git config --global user.email "jeseok@dblab.postech.ac.kr"
git config --global user.name "jeseok"
git submodule update --init --recursive

# --- Create Conda Environment from YAML file ---
# This is the standard and most reliable way to create a Conda environment.
# It handles both conda packages and pip packages listed in the yml file.
echo ">>> Creating/Updating Conda environment 'tsb-ad-env' from environment.yml..."
conda env create -f TSPulse2/environment.yml --force

echo ">>> Environment setup complete. Activating 'tsb-ad-env'..."
# The following line should be sourced in your shell or added to .bashrc to make the env active
# For scripts, you'd activate it like this:
# conda activate tsb-ad-env

# --- System-level Dependencies ---
echo ">>> Updating package list and installing system prerequisites..."
sudo apt-get update
sudo apt-get install -y wget unzip zstd

# --- Download and Unzip Datasets ---
echo ">>> Ensuring Datasets directory exists..."
mkdir -p Datasets

# --- Process TSB-AD-U ---
echo ">>> Downloading TSB-AD-U dataset..."
wget -O TSB-AD-U.zip https://www.thedatum.org/datasets/TSB-AD-U.zip
echo ">>> Unzipping TSB-AD-U dataset..."
unzip -o TSB-AD-U.zip -d Datasets
echo ">>> Removing TSB-AD-U.zip..."
rm TSB-AD-U.zip

# --- Process TSB-AD-M ---
echo ">>> Downloading TSB-AD-M dataset..."
wget -O TSB-AD-M.zip https://www.thedatum.org/datasets/TSB-AD-M.zip
echo ">>> Unzipping TSB-AD-M dataset..."
unzip -o TSB-AD-M.zip -d Datasets
echo ">>> Removing TSB-AD-M.zip..."
rm TSB-AD-M.zip

echo ">>> Setup script finished successfully."
echo ">>> To activate the environment, run: conda activate tsb-ad-env"

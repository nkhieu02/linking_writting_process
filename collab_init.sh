#!/bin/bash

# Install Kaggle CLI
pip install kaggle

# Create the Kaggle configuration directory
mkdir -p ~/.kaggle

# Copy your Kaggle API key to the configuration directory
cp kaggle.json ~/.kaggle/

# Set appropriate permissions for the API key
chmod 600 ~/.kaggle/kaggle.json

# Download the Kaggle competition data
kaggle competitions download -c linking-writing-processes-to-writing-quality

# Unzip the downloaded file to a specific directory
unzip linking-writing-processes-to-writing-quality.zip -d linking_writing_process/linking-writing-processes-to-writing-quality

# Remove the downloaded ZIP file
rm linking-writing-processes-to-writing-quality.zip

# Change to the 'linking_writing_process' directory
cd linking_writing_process

# Create 'logging' and 'checkpoints' directories
mkdir -p logging checkpoints

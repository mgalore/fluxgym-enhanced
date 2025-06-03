#!/bin/bash

echo "FluxGym Installation Script for Linux/Mac"
echo "=========================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed or not in PATH"
        echo "Please install Python 3.7+ from your package manager or https://python.org"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Python found, starting installation..."
echo

# Make the script executable
chmod +x install.py

# Run the installation script
$PYTHON_CMD install.py

echo
echo "Installation script completed."
echo "Check the output above for any errors."
echo 
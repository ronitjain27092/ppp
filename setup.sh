#!/bin/bash
# Setup script for Malware Detection XAI project (Linux/Mac)

echo ""
echo "========================================"
echo "Malware Detection XAI - Setup Script"
echo "========================================"
echo ""

# Check Python version
python3 --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Python3 not found. Please install Python 3.8 or higher."
    echo "macOS: brew install python3"
    echo "Ubuntu: sudo apt-get install python3 python3-pip python3-venv"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take 5-10 minutes..."
echo ""
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Check installation
echo ""
echo "Verifying installation..."
python -c "import tensorflow; import shap; import streamlit" && {
    echo ""
    echo "========================================"
    echo "SUCCESS! All dependencies installed."
    echo "========================================"
    echo ""
    echo "To start the application, run:"
    echo ""
    echo "source venv/bin/activate"
    echo "streamlit run app.py"
    echo ""
} || {
    echo ""
    echo "ERROR: Some dependencies failed to install."
    echo "Please check the error messages above."
    echo ""
    exit 1
}

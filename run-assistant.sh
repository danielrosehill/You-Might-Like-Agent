#!/bin/bash

# Run the You Might Like recommendation assistant
# This script checks for dependencies and runs the assistant

echo "===== You Might Like ====="
echo "AI-powered recommendation assistant"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your API keys before running again."
    exit 1
fi

# Check if I-like.md file exists
if [ ! -f "I-like.md" ]; then
    echo "Error: I-like.md file not found. This file is required."
    exit 1
fi

# Run the assistant
echo "Starting recommendation assistant..."
python recommend.py

# Deactivate virtual environment
deactivate

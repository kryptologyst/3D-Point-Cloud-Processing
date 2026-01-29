#!/usr/bin/env python3
"""Setup script for 3D Point Cloud Processing project."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up 3D Point Cloud Processing project...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("✗ Python 3.10+ is required")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    # Install pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("Warning: Pre-commit hooks installation failed, continuing...")
    
    # Create necessary directories
    directories = ["data", "checkpoints", "logs", "assets", "results"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("Warning: Some tests failed, continuing...")
    
    # Check demo
    if not run_command("python -c 'import streamlit; import open3d; import plotly'", "Checking demo dependencies"):
        return False
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: streamlit run demo/app.py")
    print("2. Train a model: python scripts/train.py")
    print("3. View the README.md for detailed documentation")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

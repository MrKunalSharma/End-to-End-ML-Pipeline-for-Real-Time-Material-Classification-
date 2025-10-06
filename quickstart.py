#!/usr/bin/env python
"""
Quick Start Script for Material Classification Pipeline
Run this to verify your installation and see the pipeline in action!
"""

import sys
import subprocess
import os

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import torch
        import cv2
        import numpy
        import pandas
        print("✓ All core dependencies installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def run_pipeline():
    """Run the complete pipeline"""
    steps = [
        ("Data Preparation", "python src/data/download_dataset_alt.py"),
        ("Preprocessing", "python src/data/preprocessing.py"),
        ("Model Training", "python train_model.py"),
        ("Model Export", "python export_model.py"),
        ("Simulation", "python run_simulation.py")
    ]
    
    print("\n" + "="*60)
    print("MATERIAL CLASSIFICATION PIPELINE - QUICK START")
    print("="*60)
    
    for step_name, command in steps:
        print(f"\n[{step_name}]")
        response = input(f"Run {step_name}? (y/n): ").lower()
        
        if response == 'y':
            print(f"Running: {command}")
            subprocess.run(command, shell=True)
        else:
            print(f"Skipped {step_name}")

if __name__ == "__main__":
    print("Welcome to the Material Classification Pipeline!")
    
    if check_requirements():
        run_pipeline()
    else:
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")

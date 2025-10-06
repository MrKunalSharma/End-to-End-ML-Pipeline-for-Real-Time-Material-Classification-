"""
Quick script to train the material classification model
"""
import sys
import os
sys.path.append(os.path.abspath('.'))

from src.models.train import main

if __name__ == "__main__":
    print("Starting model training...")
    main()

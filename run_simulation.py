"""
Main script to run the material classification simulation
"""
import sys
import os
sys.path.append(os.path.abspath('.'))

from src.deployment.conveyor_simulation import main

if __name__ == "__main__":
    print("Starting Material Classification Conveyor Simulation...")
    print("="*60)
    main()

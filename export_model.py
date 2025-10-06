"""
Script to export the trained model to ONNX and TorchScript formats
"""
import sys
import os
sys.path.append(os.path.abspath('.'))

from src.deployment.export_model import main

if __name__ == "__main__":
    print("Exporting model to deployment formats...")
    main()

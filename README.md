<div align="center">

# Material Classification Pipeline

End-to-end computer vision pipeline for real-time scrap sorting across 6 material classes.

[![Status](https://img.shields.io/badge/status-active-success)](README.md)
[![Python](https://img.shields.io/badge/python-3.9%2B-green)](README.md)
[![License](https://img.shields.io/badge/license-MIT-blue)](README.md#-license)

</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#-license)

## 🎯 Overview

This repository contains a complete ML pipeline for classifying recyclable materials into six categories: Cardboard, Glass, Metal, Paper, Plastic, and Trash. It simulates a real-time conveyor belt where items are classified on the fly, with low-confidence handling and hooks for active learning.

See the full performance analysis in [`performance_report.md`](./performance_report.md).

## ✨ Features

- **Data pipeline**: Preprocessing, augmentation, and train/val/test split
- **Transfer learning**: ResNet18 backbone with a custom classification head
- **Real-time inference**: Optimized for edge; ONNX export included
- **Conveyor simulation**: Real-time loop with configurable frame interval
- **Low-confidence detection**: Flags uncertain predictions for manual review
- **Active learning-ready**: Logs misclassifications for targeted retraining

## 📁 Project Structure

```text
material-classification-pipeline/
├── src/
│   ├── data/
│   │   ├── dataset.py                # Custom dataset class
│   │   ├── augmentation.py           # Data augmentation pipeline
│   │   └── preprocessing.py          # Data cleaning and splitting
│   ├── models/
│   │   ├── classifier.py             # Model architecture
│   │   └── train.py                  # Training logic
│   ├── deployment/
│   │   ├── export_model.py           # Model conversion to ONNX/TorchScript
│   │   ├── inference.py              # Lightweight inference engine
│   │   └── conveyor_simulation.py    # Real-time simulation
│   └── utils/
│       └── helpers.py                # Utility functions
├── data/
│   ├── raw/                          # Original dataset
│   └── processed/                    # Preprocessed splits
├── models/
│   ├── best_model.pth                # Best trained model
│   └── deployed/                     # Deployment-ready models
├── results/
│   ├── training_history.png          # Training curves
│   ├── confusion_matrix.png          # Model performance
│   └── simulation_results.csv        # Inference results
├── config.py                          # Configuration settings
├── train_model.py                     # Training entry point
├── export_model.py                    # Export entry point
└── run_simulation.py                  # Simulation entry point
```

## 🚀 Quick Start

Clone and set up the environment:

```bash
git clone https://github.com/yourusername/material-classification-pipeline.git
cd material-classification-pipeline
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate
pip install -r requirements.txt
```

## 💻 Usage

1) Prepare data
```bash
python src/data/download_dataset.py
python src/data/preprocessing.py
```

2) Train the model
```bash
python train_model.py
```

3) Export for deployment
```bash
python export_model.py
```

4) Run the conveyor simulation
```bash
python run_simulation.py
```

## 📊 Dataset

We use the TrashNet dataset (2,527 images) across six categories. Rationale:

- **Industry-relevant** classes for recycling
- **Balanced** category distribution
- **High-quality** images suitable for transfer learning
- **Open-source** and well-documented

Augmentations include random crops, flips, color jitter, and rotations to improve robustness to lighting and viewpoints.

## 🏗️ Model Architecture

- **Backbone**: ResNet18 pretrained on ImageNet
- **Head**: Dropout(0.2) → Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→6)

Why ResNet18?

- Strong transfer learning baseline
- Good accuracy/speed balance for edge deployments
- Compact model size with ONNX export

## 📈 Results

- Final test accuracy: 71.5%
- See `results/confusion_matrix.png` for class-wise performance
- See training curves in `results/training_history.png`

Detailed analysis and deployment recommendations are available in [`performance_report.md`](./performance_report.md).

## 🚢 Deployment

Supported formats:

- **ONNX** for cross-platform deployment
- **TorchScript** for PyTorch-native environments

Deployment considerations:

- Ready for quantization and batch processing
- Compatible with GPU acceleration; CPU real-time capable for smaller batches
- Low-confidence routing for human-in-the-loop QA

## 👥 Contributing

Contributions are welcome! Please open an issue or a pull request. See the checklist in [`SUBMISSION_CHECKLIST.md`](./SUBMISSION_CHECKLIST.md) for reference on expected artifacts.

## 📄 License

This project is licensed under the MIT License.
# Material Classification Pipeline for Scrap Sorting

An end-to-end machine learning pipeline for real-time material classification, designed to simulate an industrial scrap sorting system using computer vision.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Deployment](#deployment)

## 🎯 Overview

This project implements a complete ML pipeline for classifying recyclable materials into 6 categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The system simulates a real-time conveyor belt scenario where materials are classified as they pass through.

## ✨ Features

- **Data Pipeline**: Automated data preprocessing, augmentation, and train/val/test splitting
- **Transfer Learning**: ResNet18 architecture with custom classification head
- **Real-time Inference**: Optimized for edge deployment with ONNX conversion
- **Conveyor Simulation**: Simulates industrial sorting with configurable frame intervals
- **Low Confidence Detection**: Flags uncertain classifications for manual review
- **Active Learning Ready**: Collects misclassified samples for model improvement

## 📁 Project Structure



                
material-classification-pipeline/
├── src/
│ ├── data/
│ │ ├── dataset.py # Custom dataset class
│ │ ├── augmentation.py # Data augmentation pipeline
│ │ └── preprocessing.py # Data cleaning and splitting
│ ├── models/
│ │ ├── classifier.py # Model architecture
│ │ └── train.py # Training logic
│ ├── deployment/
│ │ ├── export_model.py # Model conversion to ONNX/TorchScript
│ │ ├── inference.py # Lightweight inference engine
│ │ └── conveyor_simulation.py # Real-time simulation
│ └── utils/
│ └── helpers.py # Utility functions
├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Preprocessed splits
├── models/
│ ├── best_model.pth # Best trained model
│ └── deployed/ # Deployment-ready models
├── results/
│ ├── training_history.png # Training curves
│ ├── confusion_matrix.png # Model performance
│ └── simulation_results.csv # Inference results
├── config.py # Configuration settings
├── train_model.py # Training entry point
├── export_model.py # Export entry point
└── run_simulation.py # Simulation entry point




## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/material-classification-pipeline.git
cd material-classification-pipeline


          
Create virtual environment:

          

bash


python -m venv venv
.\venv\Scripts\activate  # Windows


                
Install dependencies:

          

bash


pip install -r requirements.txt


                
💻 Usage
1. Data Preparation

          

bash


# Download and prepare dataset
python src/data/download_dataset.py
python src/data/preprocessing.py


                
2. Train Model

          

bash


python train_model.py


                
3. Export for Deployment

          

bash


python export_model.py


                
4. Run Conveyor Simulation

          

bash


python run_simulation.py


                
📊 Dataset
We use the TrashNet dataset containing 2527 images across 6 material categories:

Why TrashNet?
Industry-relevant classes for recycling applications
Balanced distribution across categories
High-quality images suitable for transfer learning
Open-source and well-documented
Data Augmentation
Random crops and flips for robustness
Color jittering to handle varying lighting conditions
Rotation and distortion for different viewing angles
🏗️ Model Architecture
Base Model: ResNet18 (pretrained on ImageNet)

Why ResNet18?
Excellent balance between accuracy and inference speed
Suitable for edge deployment
Strong transfer learning capabilities
Custom Head:

Dropout (0.2) → Linear (512→256) → ReLU → Dropout (0.2) → Linear (256→6)
📈 Results
Training Performance
Final Validation Accuracy: [Your accuracy]%
Average Inference Time: [Your time] ms
Model Size:
PyTorch: 44.8 MB
ONNX: 44.7 MB
Confusion Matrix
See results/confusion_matrix.png for detailed class-wise performance.

🚢 Deployment
Model Formats
ONNX: For cross-platform deployment
TorchScript: For PyTorch-native environments
Deployment Considerations
Edge Optimization: Model quantization ready
Batch Processing: Supports single and batch inference
Hardware Acceleration: Compatible with GPU/TPU inference
🎯 Bonus Features
Manual Override: Low-confidence predictions flagged for human review
Active Learning Pipeline: Misclassified samples logged for retraining
Real-time Monitoring: Live confidence scores and performance metrics
📝 Future Improvements
Model quantization for further size reduction
Multi-model ensemble for improved accuracy
Integration with robotic sorting hardware
Web-based monitoring dashboard
👥 Contributing
Feel free to open issues or submit pull requests for improvements!

📄 License
This project is licensed under the MIT License.

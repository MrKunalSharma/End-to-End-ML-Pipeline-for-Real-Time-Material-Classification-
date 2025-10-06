import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.helpers import setup_logging

logger = setup_logging()

def create_sample_dataset():
    """Create a sample dataset structure for testing"""
    logger.info("Creating sample dataset structure...")
    
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    data_dir = Path("data/raw")
    
    for class_name in classes:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a text file with download instructions
        with open(class_dir / "download_instructions.txt", "w") as f:
            f.write(f"Please download {class_name} images and place them in this folder.\n")
            f.write("You can use images from:\n")
            f.write("1. TrashNet dataset: https://github.com/garythung/trashnet\n")
            f.write("2. Google Images (search for '{class_name} waste' or '{class_name} recycling')\n")
            f.write("3. TACO dataset: http://tacodataset.org/\n")
            f.write("\nAim for at least 50-100 images per class.\n")
    
    logger.info("Sample dataset structure created!")
    logger.info("Please manually download images for each class.")
    
    # Create sample images for immediate testing
    create_dummy_images()

def create_dummy_images():
    """Create dummy images for immediate testing"""
    import numpy as np
    import cv2
    
    logger.info("Creating dummy images for testing...")
    
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    colors = {
        "cardboard": (139, 90, 43),  # brown
        "glass": (135, 206, 235),     # sky blue
        "metal": (192, 192, 192),     # silver
        "paper": (255, 255, 255),     # white
        "plastic": (255, 182, 193),   # pink
        "trash": (128, 128, 128)      # gray
    }
    
    for class_name in classes:
        class_dir = Path("data/raw") / class_name
        
        # Create 10 dummy images per class
        for i in range(10):
            # Create a colored image with some random noise
            img = np.ones((224, 224, 3), dtype=np.uint8) * np.array(colors[class_name], dtype=np.uint8)
            
            # Add some random shapes
            for _ in range(5):
                x, y = np.random.randint(0, 200, 2)
                radius = np.random.randint(5, 30)
                color_variation = np.random.randint(-50, 50, 3)
                color = np.clip(np.array(colors[class_name]) + color_variation, 0, 255)
                cv2.circle(img, (x, y), radius, color.tolist(), -1)
            
            # Add some noise
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            # Save image
            cv2.imwrite(str(class_dir / f"dummy_{i:03d}.jpg"), img)
    
    logger.info("Dummy images created successfully!")

def download_from_url():
    """Alternative download method"""
    logger.info("Attempting alternative download method...")
    
    # Create directories
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Alternative: Use a smaller dataset or create instructions
    logger.info("Due to download limitations, please:")
    logger.info("1. Visit https://github.com/garythung/trashnet")
    logger.info("2. Download the dataset manually")
    logger.info("3. Extract it to the data/raw folder")
    logger.info("4. Or use the dummy images created for testing")
    
    create_sample_dataset()

if __name__ == "__main__":
    download_from_url()

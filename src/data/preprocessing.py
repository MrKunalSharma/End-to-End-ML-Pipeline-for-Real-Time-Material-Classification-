import os
import sys
import cv2
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.helpers import setup_logging, load_config

logger = setup_logging()
config = load_config()

class DataPreprocessor:
    def __init__(self):
        self.config = config['data']
        self.image_size = self.config['image_size']
        self.classes = self.config['classes']
        
    def clean_dataset(self):
        """Remove corrupted images and standardize format"""
        logger.info("Cleaning dataset...")
        raw_dir = Path("data/raw")
        cleaned_count = 0
        
        # Remove system folders
        system_folders = ['__MACOSX', '.DS_Store', 'Thumbs.db']
        for folder in system_folders:
            folder_path = raw_dir / folder
            if folder_path.exists():
                if folder_path.is_dir():
                    shutil.rmtree(folder_path)
                    logger.info(f"Removed system folder: {folder}")
                else:
                    folder_path.unlink()
                    logger.info(f"Removed system file: {folder}")
        
        # Clean image files
        for class_dir in raw_dir.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.') and class_dir.name not in system_folders:
                for img_path in list(class_dir.glob("*")):
                    if img_path.is_file():
                        try:
                            # Check if it's an image file
                            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                                img_path.unlink()
                                cleaned_count += 1
                                continue
                                
                            # Try to read image
                            img = cv2.imread(str(img_path))
                            if img is None:
                                img_path.unlink()
                                cleaned_count += 1
                                logger.warning(f"Removed corrupted image: {img_path}")
                        except Exception as e:
                            img_path.unlink()
                            cleaned_count += 1
                            logger.warning(f"Removed problematic image: {img_path} - {e}")
        
        logger.info(f"Cleaned {cleaned_count} files")
    
    def create_splits(self):
        """Split dataset into train, val, and test sets"""
        logger.info("Creating train/val/test splits...")
        
        raw_dir = Path("data/raw")
        processed_dir = Path("data/processed")
        
        # Create directories
        for split in ['train', 'val', 'test']:
            for class_name in self.classes:
                (processed_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Split data
        split_info = {}
        
        for class_name in self.classes:
            class_dir = raw_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_name}")
                continue
                
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            
            if len(images) == 0:
                logger.warning(f"No images found for class: {class_name}")
                continue
            
            # First split: train+val vs test
            train_val_imgs, test_imgs = train_test_split(
                images, 
                test_size=self.config['test_split'], 
                random_state=42
            )
            
            # Second split: train vs val
            if len(train_val_imgs) > 1:
                train_imgs, val_imgs = train_test_split(
                    train_val_imgs,
                    test_size=self.config['val_split']/(1-self.config['test_split']),
                    random_state=42
                )
            else:
                train_imgs = train_val_imgs
                val_imgs = []
            
            split_info[class_name] = {
                'train': len(train_imgs),
                'val': len(val_imgs),
                'test': len(test_imgs),
                'total': len(images)
            }
            
            # Copy images to respective directories
            for img_set, split_name in [(train_imgs, 'train'), 
                                         (val_imgs, 'val'), 
                                         (test_imgs, 'test')]:
                for img_path in img_set:
                    dest_path = processed_dir / split_name / class_name / img_path.name
                    shutil.copy2(img_path, dest_path)
        
        # Save split information
        with open(processed_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=4)
        
        logger.info("Data splits created successfully!")
        for class_name, info in split_info.items():
            logger.info(f"{class_name}: Train={info['train']}, Val={info['val']}, Test={info['test']}")
    
    def preprocess_images(self):
        """Resize and normalize images"""
        logger.info("Preprocessing images...")
        processed_dir = Path("data/processed")
        
        for split in ['train', 'val', 'test']:
            split_dir = processed_dir / split
            
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
                    
                    for img_path in tqdm(image_files, desc=f"Processing {split}/{class_dir.name}"):
                        try:
                            # Read image
                            img = cv2.imread(str(img_path))
                            
                            if img is not None:
                                # Resize
                                img_resized = cv2.resize(img, (self.image_size, self.image_size))
                                
                                # Save preprocessed image
                                cv2.imwrite(str(img_path), img_resized)
                        except Exception as e:
                            logger.error(f"Error processing {img_path}: {e}")
        
        logger.info("Image preprocessing completed!")

def main():
    preprocessor = DataPreprocessor()
    
    # Clean dataset
    preprocessor.clean_dataset()
    
    # Create splits
    preprocessor.create_splits()
    
    # Preprocess images
    preprocessor.preprocess_images()

if __name__ == "__main__":
    main()

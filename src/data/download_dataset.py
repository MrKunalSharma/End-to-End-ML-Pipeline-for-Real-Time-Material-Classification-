import os
import zipfile
import requests
from pathlib import Path
import shutil
from tqdm import tqdm
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.helpers import setup_logging, load_config

logger = setup_logging()

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file:
        with tqdm(total=total_size, unit='iB', unit_scale=True) as progress_bar:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                file.write(data)

def download_trashnet():
    """Download TrashNet dataset"""
    # Using a direct link to TrashNet dataset
    dataset_url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_dir / "dataset-resized.zip"
    
    if not zip_path.exists():
        logger.info("Downloading TrashNet dataset...")
        try:
            download_file(dataset_url, zip_path)
            logger.info("Download completed!")
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            # Alternative: manual download instruction
            logger.info("Please download manually from: https://github.com/garythung/trashnet")
            return False
    
    # Extract dataset
    logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Reorganize directory structure
    dataset_path = data_dir / "dataset-resized"
    if dataset_path.exists():
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                target_dir = data_dir / class_dir.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(class_dir), str(data_dir))
        
        # Remove empty directory and zip
        shutil.rmtree(dataset_path)
        os.remove(zip_path)
    
    logger.info("Dataset preparation completed!")
    return True

if __name__ == "__main__":
    success = download_trashnet()
    if success:
        # Count images per class
        data_dir = Path("data/raw")
        for class_dir in sorted(data_dir.iterdir()):
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                image_count = len(list(class_dir.glob("*.jpg")))
                logger.info(f"{class_dir.name}: {image_count} images")

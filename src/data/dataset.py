import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.utils.helpers import load_config

config = load_config()

class MaterialDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (str): Path to data directory
            split (str): 'train', 'val', or 'test'
            transform: Albumentations transform
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = config['data']['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
    
    def get_sample_weights(self):
        """Calculate sample weights for balanced sampling"""
        class_counts = np.zeros(len(self.classes))
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Calculate weights
        weights = 1.0 / class_counts
        sample_weights = [weights[label] for _, label in self.samples]
        
        return torch.DoubleTensor(sample_weights)

def get_data_loaders(data_dir='data/processed', batch_size=None):
    """Create data loaders for train, val, and test sets"""
    if batch_size is None:
        batch_size = config['data']['batch_size']
    
    # Create datasets
    train_dataset = MaterialDataset(
        data_dir, 
        split='train', 
        transform=get_train_transforms()
    )
    
    val_dataset = MaterialDataset(
        data_dir, 
        split='val', 
        transform=get_val_transforms()
    )
    
    test_dataset = MaterialDataset(
        data_dir, 
        split='test', 
        transform=get_val_transforms()
    )
    
    # Create weighted sampler for balanced training
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_dataset.get_sample_weights(),
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Check one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")

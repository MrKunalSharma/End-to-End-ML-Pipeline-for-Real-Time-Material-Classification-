import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.helpers import load_config

config = load_config()

def get_train_transforms():
    """Get augmentation transforms for training data"""
    aug_config = config['augmentation']['train']
    image_size = config['data']['image_size']
    
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),  # Changed from height/width to size tuple
            scale=(0.8, 1.0)
        ),
        A.HorizontalFlip(p=aug_config['horizontal_flip']),
        A.VerticalFlip(p=aug_config['vertical_flip']),
        A.Rotate(limit=aug_config['rotation_degree'], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=aug_config['brightness_contrast'],
            contrast_limit=aug_config['brightness_contrast'],
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3),
        ], p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms():
    """Get transforms for validation/test data"""
    image_size = config['data']['image_size']
    
    return A.Compose([
        A.Resize(
            height=image_size, 
            width=image_size
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def visualize_augmentations(image_path, num_samples=5):
    """Visualize augmentations on a single image"""
    import matplotlib.pyplot as plt
    
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_train_transforms()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Augmented versions
    for i in range(1, min(num_samples + 1, 6)):
        augmented = transform(image=image)['image']
        # Denormalize for visualization
        augmented = augmented.permute(1, 2, 0).numpy()
        augmented = (augmented * np.array([0.229, 0.224, 0.225]) + 
                    np.array([0.485, 0.456, 0.406]))
        augmented = np.clip(augmented, 0, 1)
        
        axes[i].imshow(augmented)
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/augmentation_samples.png")
    plt.close()

if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentations...")
    
    # Find a sample image
    sample_image = None
    for class_dir in Path("data/processed/train").iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            if images:
                sample_image = images[0]
                break
    
    if sample_image:
        visualize_augmentations(sample_image)
        print("Augmentation samples saved to results/augmentation_samples.png")

﻿# Configuration settings
config = {
    'project': {
        'name': 'Material Classification Pipeline',
        'version': '1.0.0'
    },
    'data': {
        'dataset_name': 'TrashNet',
        'classes': ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
        'image_size': 224,
        'batch_size': 32,
        'num_workers': 0,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15
    },
    'model': {
        'architecture': 'resnet18',
        'pretrained': True,
        'num_classes': 6,
        'learning_rate': 0.001,
        'epochs': 10,
        'early_stopping_patience': 3
    },
    'augmentation': {
        'train': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.2,
            'rotation_degree': 20,
            'brightness_contrast': 0.2
        }
    },
    'deployment': {
        'confidence_threshold': 0.7,
        'inference_device': 'cpu',
        'onnx_export': True,
        'torchscript_export': True
    },
    'simulation': {
        'frame_interval': 2,
        'log_low_confidence': True,
        'enable_manual_override': True
    }
}

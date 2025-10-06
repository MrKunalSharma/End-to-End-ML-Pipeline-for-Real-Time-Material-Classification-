import os
import json
from datetime import datetime
import logging
from pathlib import Path

def load_config(config_path='config.py'):
    """Load configuration from Python file"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def setup_logging(log_dir='logs'):
    """Setup logging configuration"""
    Path(log_dir).mkdir(exist_ok=True)
    log_file = os.path.join(log_dir, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directories():
    """Ensure all required directories exist"""
    dirs = [
        'data/raw', 'data/processed', 'data/augmented',
        'models', 'results', 'logs', 'notebooks'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def save_metrics(metrics, filepath):
    """Save metrics to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
        
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

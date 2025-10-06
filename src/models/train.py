import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import get_data_loaders
from src.models.classifier import create_model
from src.utils.helpers import setup_logging, load_config, save_metrics, AverageMeter

config = load_config()
logger = setup_logging()

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['model']['learning_rate']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=2, 
            factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0
        self.epochs_without_improvement = 0
        
    def train_epoch(self):
        self.model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            acc = (predicted == targets).float().mean()
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.4f}'
            })
        
        return losses.avg, accuracies.avg
    
    def validate(self):
        self.model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                _, predicted = outputs.max(1)
                acc = (predicted == targets).float().mean()
                
                losses.update(loss.item(), images.size(0))
                accuracies.update(acc.item(), images.size(0))
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc': f'{accuracies.avg:.4f}'
                })
        
        return losses.avg, accuracies.avg, all_predictions, all_targets
    
    def train(self, epochs=None):
        if epochs is None:
            epochs = config['model']['epochs']
        
        logger.info(f"Training on device: {self.device}")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, _, _ = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= config['model']['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pth', epochs, val_acc)
        
        # Plot training history
        self.plot_training_history()
        
        return self.history
    
    def save_checkpoint(self, filename, epoch, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': val_acc,
            'config': config,
            'history': self.history
        }
        
        save_path = Path('models') / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png')
        plt.close()

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set and generate detailed metrics"""
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    
    # Classification report
    classes = config['data']['classes']
    report = classification_report(
        all_targets, 
        all_predictions, 
        target_names=classes,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    save_metrics(metrics, 'results/test_metrics.json')
    
    return metrics

def main():
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Create model
    model = create_model()
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader)
    
    # Train model
    history = trainer.train()
    
    # Load best model
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader)
    
    logger.info(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    logger.info("\nClassification Report:")
    for class_name, class_metrics in metrics['classification_report'].items():
        if isinstance(class_metrics, dict):
            logger.info(f"{class_name}: precision={class_metrics['precision']:.3f}, "
                       f"recall={class_metrics['recall']:.3f}, "
                       f"f1-score={class_metrics['f1-score']:.3f}")

if __name__ == "__main__":
    main()

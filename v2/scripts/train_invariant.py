"""
Training script for hand-invariant posture classification.
Includes consistency regularization loss for improved generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import time

from ..config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    CLASSIFICATION_LOSS_WEIGHT,
    CONSISTENCY_LOSS_WEIGHT,
    CONSISTENCY_TEMPERATURE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_MIN_LR,
    EARLY_STOPPING_PATIENCE,
    NUM_WORKERS,
    PIN_MEMORY,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    get_device,
    get_model_save_path
)
from ..models.hybrid_model import create_model
from ..data.datasets import create_dataloaders
from ..utils.transforms import get_consistency_transform


class ConsistencyLoss(nn.Module):
    """
    Consistency regularization loss.
    Enforces that predictions remain stable under spatial transformations.
    """
    
    def __init__(self, temperature: float = CONSISTENCY_TEMPERATURE):
        """
        Args:
            temperature: Temperature for softening probability distributions
        """
        super().__init__()
        self.temperature = temperature
        self.consistency_transform = get_consistency_transform()
    
    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        pose_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss.
        
        Args:
            model: The posture classification model
            images: Original images (B, 3, 224, 224)
            pose_features: Pose features (B, 24)
            
        Returns:
            Consistency loss value
        """
        # Get original predictions (detached to prevent gradient flow)
        with torch.no_grad():
            original_logits = model(images, pose_features)
            original_probs = F.softmax(original_logits / self.temperature, dim=1)
        
        # Apply spatial transformation (vertical shift)
        augmented_images = torch.stack([
            self.consistency_transform(img) for img in images
        ])
        
        # Get augmented predictions (with gradient)
        augmented_logits = model(augmented_images, pose_features)
        augmented_log_probs = F.log_softmax(augmented_logits / self.temperature, dim=1)
        
        # KL divergence between original and augmented distributions
        consistency_loss = F.kl_div(
            augmented_log_probs,
            original_probs,
            reduction='batchmean'
        )
        
        return consistency_loss


class Trainer:
    """
    Trainer class for posture classification with consistency regularization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY
    ):
        """
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.consistency_criterion = ConsistencyLoss()
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=LR_SCHEDULER_MIN_LR,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_cls_loss': [],
            'train_cons_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_cons_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        
        for batch_idx, (images, pose_features, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            pose_features = pose_features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images, pose_features)
            
            # Classification loss
            cls_loss = self.classification_criterion(logits, labels)
            
            # Consistency loss
            cons_loss = self.consistency_criterion(self.model, images, pose_features)
            
            # Combined loss
            loss = (CLASSIFICATION_LOSS_WEIGHT * cls_loss + 
                   CONSISTENCY_LOSS_WEIGHT * cons_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_cons_loss += cons_loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'cons': f'{cons_loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_cons_loss = total_cons_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'cons_loss': avg_cons_loss,
            'accuracy': accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for images, pose_features, labels in pbar:
                # Move to device
                images = images.to(self.device)
                pose_features = pose_features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images, pose_features)
                
                # Loss (classification only for validation)
                loss = self.classification_criterion(logits, labels)
                total_loss += loss.item()
                
                # Accuracy
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(self, num_epochs: int = NUM_EPOCHS):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_cls_loss'].append(train_metrics['cls_loss'])
            self.history['train_cons_loss'].append(train_metrics['cons_loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Cls: {train_metrics['cls_loss']:.4f}, "
                  f"Cons: {train_metrics['cons_loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                
                # Save checkpoint
                self.save_checkpoint(get_model_save_path(best=True), epoch, val_metrics)
                print(f"  ✓ New best model! Val Acc: {self.best_val_acc:.2f}%")
            else:
                self.epochs_no_improve += 1
            
            # Early stopping
            if self.epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print("-"*80)
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch+1})")
        print("="*80)
        
        # Save final model
        self.save_checkpoint(get_model_save_path(best=False), num_epochs-1, val_metrics)
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Current metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics': metrics,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved to {path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train Total Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(epochs, self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Component losses
        axes[1, 0].plot(epochs, self.history['train_cls_loss'], label='Classification Loss')
        axes[1, 0].plot(epochs, self.history['train_cons_loss'], label='Consistency Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(epochs, self.history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        save_path = RESULTS_DIR / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nTraining curves saved to {save_path}")
        plt.close()


def main():
    """Main training function."""
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(device=device)
    
    # Create dataloaders
    print("\nLoading datasets...")
    # Note: Update these paths to your actual data directories
    train_dir = "path/to/train"
    val_dir = "path/to/val"
    
    dataloaders = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=device
    )
    
    # Train model
    trainer.train()


if __name__ == "__main__":
    main()
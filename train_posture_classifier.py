"""
train_posture_classifier.py
Train a CNN classifier for sitting posture recognition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import (
    POSTURE_CLASSES, CLASSIFIER_CONFIG, POSTURE_DATASET_PATH,
    POSTURE_CLASSIFIER_PATH, RESULTS_ROOT
)
from datasets import PostureDataset, get_posture_transforms


class PostureClassifier(nn.Module):
    """
    CNN classifier for posture recognition.
    Uses a pretrained backbone (ResNet or MobileNet) with custom classifier head.
    """
    
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True):
        """
        Args:
            num_classes: Number of posture classes
            backbone: Backbone architecture (resnet18, resnet34, mobilenet_v3_small)
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(PostureClassifier, self).__init__()
        
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original FC layer
        
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        elif backbone == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path: Path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")


def train_posture_classifier():
    """Main training function for posture classifier."""
    print(f"\n{'='*60}")
    print("Training Posture Classifier")
    print(f"{'='*60}\n")
    
    # Create results directory
    results_dir = RESULTS_ROOT / "posture_training"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(CLASSIFIER_CONFIG["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_transform = get_posture_transforms("train")
    val_transform = get_posture_transforms("val")
    
    train_dataset = PostureDataset(
        POSTURE_DATASET_PATH,
        split="train",
        transform=train_transform
    )
    
    val_dataset = PostureDataset(
        POSTURE_DATASET_PATH,
        split="val",
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CLASSIFIER_CONFIG["batch_size"],
        shuffle=True,
        num_workers=CLASSIFIER_CONFIG["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CLASSIFIER_CONFIG["batch_size"],
        shuffle=False,
        num_workers=CLASSIFIER_CONFIG["num_workers"],
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {CLASSIFIER_CONFIG['batch_size']}\n")
    
    # Initialize model
    print(f"Initializing {CLASSIFIER_CONFIG['backbone']} model...")
    model = PostureClassifier(
        num_classes=len(POSTURE_CLASSES),
        backbone=CLASSIFIER_CONFIG["backbone"],
        pretrained=CLASSIFIER_CONFIG["pretrained"]
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CLASSIFIER_CONFIG["learning_rate"],
        weight_decay=CLASSIFIER_CONFIG["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(CLASSIFIER_CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CLASSIFIER_CONFIG['epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, POSTURE_CLASSIFIER_PATH)
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*60 + "\n")
    
    # Plot training history
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        results_dir / "training_history.png"
    )
    
    # Load best model and generate final evaluation
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(POSTURE_CLASSIFIER_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds, POSTURE_CLASSES,
        results_dir / "confusion_matrix.png"
    )
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=POSTURE_CLASSES))
    
    print(f"\nModel saved to: {POSTURE_CLASSIFIER_PATH}")
    print(f"Results saved to: {results_dir}")


# Main script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train posture classifier")
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.epochs:
        CLASSIFIER_CONFIG["epochs"] = args.epochs
    if args.batch_size:
        CLASSIFIER_CONFIG["batch_size"] = args.batch_size
    if args.lr:
        CLASSIFIER_CONFIG["learning_rate"] = args.lr
    
    # Check CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Train
    train_posture_classifier()
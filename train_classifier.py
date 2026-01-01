"""
train_classifier.py
Simplified training script for your posture classification dataset.
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
    POSTURE_CLASSES, CLASSIFIER_CONFIG, YOLO_DATASET_PATH,
    POSTURE_CLASSIFIER_PATH, RESULTS_ROOT, POSTURE_DISPLAY_NAMES
)
from datasets import PostureDataset, get_posture_transforms


class PostureClassifier(nn.Module):
    """CNN classifier for posture recognition."""
    
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True):
        super(PostureClassifier, self).__init__()
        
        self.backbone_name = backbone
        
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
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
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
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
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Use display names if available
    display_labels = [POSTURE_DISPLAY_NAMES.get(cls, cls) for cls in class_names]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_labels, yticklabels=display_labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Confusion matrix saved to {save_path}")


def main():
    """Main training function."""
    print("\n" + "="*70)
    print(" " * 15 + "GAMER ERGOVISION - POSTURE CLASSIFIER TRAINING")
    print("="*70 + "\n")
    
    # Create results directory
    results_dir = RESULTS_ROOT / "posture_training"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = torch.device(CLASSIFIER_CONFIG["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_transform = get_posture_transforms("train")
    val_transform = get_posture_transforms("val")
    
    train_dataset = PostureDataset(YOLO_DATASET_PATH, split="train", transform=train_transform)
    val_dataset = PostureDataset(YOLO_DATASET_PATH, split="val", transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CLASSIFIER_CONFIG["batch_size"],
        shuffle=True,
        num_workers=CLASSIFIER_CONFIG["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CLASSIFIER_CONFIG["batch_size"],
        shuffle=False,
        num_workers=CLASSIFIER_CONFIG["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataset Summary:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {CLASSIFIER_CONFIG['batch_size']}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}\n")
    
    # Initialize model
    print(f"Initializing {CLASSIFIER_CONFIG['backbone']} model...")
    model = PostureClassifier(
        num_classes=len(POSTURE_CLASSES),
        backbone=CLASSIFIER_CONFIG["backbone"],
        pretrained=CLASSIFIER_CONFIG["pretrained"]
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CLASSIFIER_CONFIG["learning_rate"],
        weight_decay=CLASSIFIER_CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(CLASSIFIER_CONFIG["epochs"]):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CLASSIFIER_CONFIG['epochs']}")
        print('='*70)
        
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
        
        # Print summary
        print(f"\nEpoch {epoch+1} Results:")
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
                'train_acc': train_acc,
                'train_loss': train_loss,
            }, POSTURE_CLASSIFIER_PATH)
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Training complete
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {POSTURE_CLASSIFIER_PATH}\n")
    
    # Plot training history
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        results_dir / "training_history.png"
    )
    
    # Final evaluation on best model
    print("\n" + "="*70)
    print("Final Evaluation on Best Model")
    print("="*70 + "\n")
    
    checkpoint = torch.load(POSTURE_CLASSIFIER_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
    
    # Confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds, POSTURE_CLASSES,
        results_dir / "confusion_matrix.png"
    )
    
    # Classification report
    display_labels = [POSTURE_DISPLAY_NAMES.get(cls, cls) for cls in POSTURE_CLASSES]
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=display_labels))
    
    # Save report to file
    report_path = results_dir / "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("GAMER ERGOVISION - CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n\n")
        f.write(classification_report(val_labels, val_preds, target_names=display_labels))
    print(f"\n✓ Report saved to {report_path}")
    
    print("\n" + "="*70)
    print("All results saved to:", results_dir)
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train posture classifier")
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of epochs (default: {CLASSIFIER_CONFIG["epochs"]})')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Batch size (default: {CLASSIFIER_CONFIG["batch_size"]})')
    parser.add_argument('--lr', type=float, default=None,
                       help=f'Learning rate (default: {CLASSIFIER_CONFIG["learning_rate"]})')
    parser.add_argument('--backbone', type=str, default=None,
                       choices=['resnet18', 'resnet34', 'mobilenet_v3_small'],
                       help=f'Backbone architecture (default: {CLASSIFIER_CONFIG["backbone"]})')
    
    args = parser.parse_args()
    
    # Override config
    if args.epochs:
        CLASSIFIER_CONFIG["epochs"] = args.epochs
    if args.batch_size:
        CLASSIFIER_CONFIG["batch_size"] = args.batch_size
    if args.lr:
        CLASSIFIER_CONFIG["learning_rate"] = args.lr
    if args.backbone:
        CLASSIFIER_CONFIG["backbone"] = args.backbone
    
    # Check CUDA
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    # Run training
    main()
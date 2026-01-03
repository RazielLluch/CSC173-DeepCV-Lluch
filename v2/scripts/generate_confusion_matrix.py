"""
Standalone Confusion Matrix Generator
Creates beautiful confusion matrix visualizations
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ..config import (
    POSTURE_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    RESULTS_DIR,
    get_device,
    get_model_save_path
)
from ..models.hybrid_model import create_model
from ..data.datasets import InvariantPostureDataset
from torch.utils.data import DataLoader


def generate_confusion_matrix(
    test_dir: str,
    checkpoint_path: Path = None,
    output_path: Path = None,
    figsize: tuple = (14, 6),
    cmap: str = 'Blues',
    normalize: bool = True
):
    """
    Generate and save confusion matrix visualization.
    
    Args:
        test_dir: Path to test data directory
        checkpoint_path: Path to model checkpoint
        output_path: Where to save the plot
        figsize: Figure size (width, height)
        cmap: Colormap for heatmap
        normalize: Whether to show normalized version
    """
    
    print("="*70)
    print("CONFUSION MATRIX GENERATOR")
    print("="*70)
    
    # Setup
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Load model
    print("Loading model...")
    model = create_model(device=device)
    
    if checkpoint_path is None:
        checkpoint_path = get_model_save_path(best=True)
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"⚠️  Warning: No checkpoint found at {checkpoint_path}")
        print("   Using random weights (results will be meaningless)")
    
    model.eval()
    
    # Load test data
    print(f"\nLoading test data from: {test_dir}")
    test_dataset = InvariantPostureDataset(
        test_dir,
        split='test',
        extract_pose_features=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    print(f"Test set size: {len(test_dataset)} samples")
    
    # Run inference
    print("\nRunning inference...")
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, pose_features, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            pose_features = pose_features.to(device)
            
            logits = model(images, pose_features)
            _, predictions = torch.max(logits, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Compute confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create nicer class names
    class_display_names = {
        'backwardbadposture': 'Backward Bad',
        'forwardbadposture': 'Forward Bad',
        'goodposture': 'Good Posture'
    }
    display_names = [class_display_names.get(cls, cls) for cls in POSTURE_CLASSES]
    
    # Create visualization
    if normalize:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Absolute counts
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            xticklabels=display_names,
            yticklabels=display_names,
            ax=axes[0],
            cbar_kws={'label': 'Count'},
            square=True,
            linewidths=0.5,
            linecolor='gray'
        )
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap=cmap,
            xticklabels=display_names,
            yticklabels=display_names,
            ax=axes[1],
            cbar_kws={'label': 'Percentage'},
            square=True,
            linewidths=0.5,
            linecolor='gray',
            vmin=0,
            vmax=1
        )
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            xticklabels=display_names,
            yticklabels=display_names,
            ax=ax,
            cbar_kws={'label': 'Count'},
            square=True,
            linewidths=0.5,
            linecolor='gray'
        )
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        RESULTS_DIR.mkdir(exist_ok=True)
        output_path = RESULTS_DIR / 'confusion_matrix.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to: {output_path}")
    
    # Print statistics
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\n{'='*70}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*70}")
    print("\nPer-Class Metrics:")
    for i, name in enumerate(display_names):
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{name}:")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall: {recall*100:.2f}%")
        print(f"  F1-Score: {f1*100:.2f}%")
        print(f"  Support: {cm[i, :].sum()}")
    
    print(f"\n{'='*70}\n")
    
    # Show plot
    try:
        plt.show()
    except:
        pass  # In case display is not available
    
    plt.close()
    
    return cm, cm_normalized


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix visualization'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        required=True,
        help='Path to test data directory'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for confusion matrix (default: results/confusion_matrix.png)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Only show counts (no normalized version)'
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default='Blues',
        choices=['Blues', 'Greens', 'Reds', 'Purples', 'YlOrRd', 'viridis'],
        help='Color map for heatmap'
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    output_path = Path(args.output) if args.output else None
    
    generate_confusion_matrix(
        test_dir=args.test_dir,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        normalize=not args.no_normalize,
        cmap=args.cmap
    )


if __name__ == "__main__":
    main()
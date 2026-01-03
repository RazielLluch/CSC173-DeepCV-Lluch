"""
Enhanced Test Evaluation Script
Produces formatted classification report with metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from ..config import (
    POSTURE_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    get_device,
    get_model_save_path
)
from ..models.hybrid_model import create_model
from ..data.datasets import InvariantPostureDataset
from torch.utils.data import DataLoader


def evaluate_model(test_dir: str, checkpoint_path: Path = None):
    """
    Evaluate model and print formatted report.
    
    Args:
        test_dir: Path to test data directory
        checkpoint_path: Path to model checkpoint (optional)
    """
    
    # Set device
    device = get_device()
    
    # Load model
    print("Loading model...")
    model = create_model(device=device)
    
    if checkpoint_path is None:
        checkpoint_path = get_model_save_path(best=True)
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        if 'best_val_acc' in checkpoint:
            print(f"   Validation accuracy: {checkpoint['best_val_acc']:.2f}%\n")
    else:
        print(f"⚠️  Warning: Checkpoint not found at {checkpoint_path}\n")
    
    model.eval()
    
    # Load test dataset
    print("Loading test dataset...")
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
    
    print(f"Test set loaded: {len(test_dataset)} samples\n")
    
    # Run evaluation
    print("Evaluating on test set...\n")
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, pose_features, labels in tqdm(test_loader, desc="Testing"):
            # Move to device
            images = images.to(device)
            pose_features = pose_features.to(device)
            
            # Forward pass
            logits = model(images, pose_features)
            probabilities = F.softmax(logits, dim=1)
            _, predictions = torch.max(logits, 1)
            
            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Create class name mapping with better formatting
    class_display_names = {
        'backwardbadposture': 'Reclined/Backward',
        'forwardbadposture': 'Forward Lean',
        'goodposture': 'Neutral/Good'
    }
    
    display_names = [class_display_names.get(cls, cls) for cls in POSTURE_CLASSES]
    
    # Get per-class counts
    unique, counts = np.unique(all_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # Print formatted report
    print("\n")
    print("="*70)
    print("GAMER ERGOVISION - TEST SET CLASSIFICATION REPORT")
    print("="*70)
    print()
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Total Test Samples: {len(all_labels)}")
    print()
    print("Classification Report:")
    
    # Generate classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=display_names,
        digits=2
    )
    print(report)
    
    print("="*70)
    print("Per-Class Sample Counts:")
    for i, cls in enumerate(POSTURE_CLASSES):
        display_name = class_display_names.get(cls, cls)
        count = class_counts.get(i, 0)
        print(f"  {display_name}: {count}")
    print()
    
    # Calculate and display confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    print("="*70)
    print("Confusion Matrix:")
    print()
    
    # Header
    print("                  ", end="")
    for name in display_names:
        print(f"{name:>17}", end=" ")
    print()
    print("-" * 70)
    
    # Rows
    for i, true_name in enumerate(display_names):
        print(f"{true_name:>17}", end=" ")
        for j in range(len(display_names)):
            print(f"{cm[i][j]:>17}", end=" ")
        print()
    
    print("="*70)
    print()
    
    # Save detailed report to file
    from config import RESULTS_DIR
    RESULTS_DIR.mkdir(exist_ok=True)
    
    report_path = RESULTS_DIR / 'test_evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GAMER ERGOVISION - TEST SET CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Total Test Samples: {len(all_labels)}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n" + "="*70 + "\n")
        f.write("Per-Class Sample Counts:\n")
        for i, cls in enumerate(POSTURE_CLASSES):
            display_name = class_display_names.get(cls, cls)
            count = class_counts.get(i, 0)
            f.write(f"  {display_name}: {count}\n")
        f.write("\n")
    
    print(f"✅ Detailed report saved to: {report_path}")
    
    return accuracy, all_labels, all_predictions, all_probabilities


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate posture classifier on test set'
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
        help='Path to model checkpoint (default: checkpoints/best_model.pth)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    evaluate_model(args.test_dir, checkpoint_path)


if __name__ == "__main__":
    main()
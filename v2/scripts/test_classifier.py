"""
Comprehensive test script for posture classifier.
Generates confusion matrix, classification report, and per-class metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

from config import (
    POSTURE_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    RESULTS_DIR,
    get_device,
    get_model_save_path
)
from models.hybrid_model import create_model
from data.datasets import create_dataloaders


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to run evaluation on
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        
        self.model.eval()
    
    def evaluate(self):
        """
        Run evaluation and collect predictions.
        
        Returns:
            Tuple of (all_labels, all_predictions, all_probabilities)
        """
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        print("Evaluating model on test set...")
        
        with torch.no_grad():
            for images, pose_features, labels in tqdm(self.test_loader, desc="Testing"):
                # Move to device
                images = images.to(self.device)
                pose_features = pose_features.to(self.device)
                
                # Forward pass
                logits = self.model(images, pose_features)
                probabilities = F.softmax(logits, dim=1)
                _, predictions = torch.max(logits, 1)
                
                # Collect results
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return (
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
    
    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        save_path: Path
    ):
        """
        Plot and save confusion matrix.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            save_path: Path to save plot
        """
        # Compute confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute counts
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=POSTURE_CLASSES,
            yticklabels=POSTURE_CLASSES,
            ax=axes[0]
        )
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Normalized
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=POSTURE_CLASSES,
            yticklabels=POSTURE_CLASSES,
            ax=axes[1]
        )
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_per_class_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        save_path: Path
    ):
        """
        Plot per-class precision, recall, and F1-score.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            save_path: Path to save plot
        """
        # Compute metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(len(POSTURE_CLASSES)))
        )
        
        # Create bar plot
        x = np.arange(len(POSTURE_CLASSES))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Posture Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(POSTURE_CLASSES, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', fontsize=9)
            ax.text(i, r + 0.02, f'{r:.2f}', ha='center', fontsize=9)
            ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class metrics plot saved to {save_path}")
        plt.close()
    
    def plot_probability_distributions(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        save_path: Path
    ):
        """
        Plot probability distribution for correct vs incorrect predictions.
        
        Args:
            labels: True labels
            probabilities: Predicted probabilities
            save_path: Path to save plot
        """
        # Get max probabilities
        max_probs = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        
        # Separate correct and incorrect
        correct_mask = predictions == labels
        correct_probs = max_probs[correct_mask]
        incorrect_probs = max_probs[~correct_mask]
        
        # Plot distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(correct_probs, bins=50, alpha=0.7, label='Correct', color='green')
        ax.hist(incorrect_probs, bins=50, alpha=0.7, label='Incorrect', color='red')
        
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Correct: μ={np.mean(correct_probs):.3f}, σ={np.std(correct_probs):.3f}\n"
        stats_text += f"Incorrect: μ={np.mean(incorrect_probs):.3f}, σ={np.std(incorrect_probs):.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Probability distribution plot saved to {save_path}")
        plt.close()
    
    def generate_report(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        save_dir: Path
    ):
        """
        Generate comprehensive evaluation report.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            probabilities: Predicted probabilities
            save_dir: Directory to save report files
        """
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)
        
        # Overall accuracy
        accuracy = accuracy_score(labels, predictions)
        print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            labels,
            predictions,
            target_names=POSTURE_CLASSES,
            digits=4
        ))
        
        # Per-class accuracy
        print("\nPer-Class Accuracy:")
        for i, class_name in enumerate(POSTURE_CLASSES):
            class_mask = labels == i
            class_acc = accuracy_score(
                labels[class_mask],
                predictions[class_mask]
            )
            print(f"  {class_name}: {class_acc*100:.2f}%")
        
        # Confidence statistics
        max_probs = np.max(probabilities, axis=1)
        correct_mask = predictions == labels
        
        print("\nConfidence Statistics:")
        print(f"  Overall - Mean: {np.mean(max_probs):.3f}, Std: {np.std(max_probs):.3f}")
        print(f"  Correct - Mean: {np.mean(max_probs[correct_mask]):.3f}, "
              f"Std: {np.std(max_probs[correct_mask]):.3f}")
        print(f"  Incorrect - Mean: {np.mean(max_probs[~correct_mask]):.3f}, "
              f"Std: {np.std(max_probs[~correct_mask]):.3f}")
        
        print("="*80)
        
        # Generate plots
        self.plot_confusion_matrix(
            labels,
            predictions,
            save_dir / 'confusion_matrix.png'
        )
        
        self.plot_per_class_metrics(
            labels,
            predictions,
            save_dir / 'per_class_metrics.png'
        )
        
        self.plot_probability_distributions(
            labels,
            probabilities,
            save_dir / 'confidence_distribution.png'
        )
        
        # Save detailed results to text file
        report_path = save_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GAMER ERGOVISION - EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
            
            f.write("Classification Report:\n")
            f.write(classification_report(
                labels,
                predictions,
                target_names=POSTURE_CLASSES,
                digits=4
            ))
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\nDetailed report saved to {report_path}")


def main():
    """
    Main evaluation function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate posture classifier')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = create_model(device=device)
    
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_model_save_path(best=True)
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    # Create test dataloader
    print("\nLoading test dataset...")
    from data.datasets import InvariantPostureDataset
    from torch.utils.data import DataLoader
    
    test_dataset = InvariantPostureDataset(
        args.test_dir,
        split='test',
        extract_pose_features=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Test set size: {len(test_dataset)} samples")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader, device)
    
    # Run evaluation
    labels, predictions, probabilities = evaluator.evaluate()
    
    # Generate report
    evaluator.generate_report(labels, predictions, probabilities, RESULTS_DIR)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
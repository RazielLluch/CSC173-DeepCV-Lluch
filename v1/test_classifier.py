"""
test_classifier.py
Test and evaluate the trained posture classifier.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    POSTURE_CLASSES, CLASSIFIER_CONFIG, YOLO_DATASET_PATH,
    POSTURE_CLASSIFIER_PATH, RESULTS_ROOT, POSTURE_DISPLAY_NAMES,
    POSTURE_RISK_MAPPING
)
from datasets import PostureDataset, get_posture_transforms
from train_classifier import PostureClassifier


def test_model(model, dataloader, device):
    """Test model and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_confidences)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    display_labels = [POSTURE_DISPLAY_NAMES.get(cls, cls) for cls in class_names]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_labels, yticklabels=display_labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - Test Set', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Confusion matrix saved to {save_path}")


def plot_confidence_distribution(confidences, predictions, labels, save_path: Path):
    """Plot confidence distribution for correct and incorrect predictions."""
    correct_mask = predictions == labels
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
    ax1.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot([correct_conf, incorrect_conf], labels=['Correct', 'Incorrect'])
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Confidence by Prediction Correctness')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Confidence distribution saved to {save_path}")


def analyze_per_class_performance(y_true, y_pred, confidences, class_names, save_path: Path):
    """Analyze and visualize per-class performance."""
    display_labels = [POSTURE_DISPLAY_NAMES.get(cls, cls) for cls in class_names]
    
    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 4))
    if len(class_names) == 1:
        axes = [axes]
    
    for i, (class_name, display_name) in enumerate(zip(class_names, display_labels)):
        class_idx = i
        class_mask = y_true == class_idx
        
        if class_mask.sum() == 0:
            continue
        
        class_preds = y_pred[class_mask]
        class_true = y_true[class_mask]
        class_conf = confidences[class_mask]
        
        # Calculate metrics
        correct = (class_preds == class_true).sum()
        total = len(class_true)
        accuracy = correct / total * 100
        avg_conf = class_conf.mean()
        
        # Plot confidence distribution for this class
        correct_mask = class_preds == class_true
        correct_conf = class_conf[correct_mask]
        incorrect_conf = class_conf[~correct_mask]
        
        axes[i].hist(correct_conf, bins=15, alpha=0.7, label='Correct', color='green')
        if len(incorrect_conf) > 0:
            axes[i].hist(incorrect_conf, bins=15, alpha=0.7, label='Incorrect', color='red')
        
        axes[i].set_title(f'{display_name}\nAcc: {accuracy:.1f}%')
        axes[i].set_xlabel('Confidence')
        axes[i].set_ylabel('Count')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Per-class analysis saved to {save_path}")


def generate_risk_analysis(y_pred, class_names, save_path: Path):
    """Analyze risk distribution based on predictions."""
    risk_counts = {'low': 0, 'medium': 0, 'high': 0}
    
    for pred_idx in y_pred:
        class_name = class_names[pred_idx]
        risk = POSTURE_RISK_MAPPING.get(class_name, 'medium')
        risk_counts[risk] += 1
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    risks = list(risk_counts.keys())
    counts = list(risk_counts.values())
    colors = ['green', 'orange', 'red']
    
    ax1.bar(risks, counts, color=colors, alpha=0.7)
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Ergonomic Risk Distribution (Test Set)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Pie chart
    ax2.pie(counts, labels=risks, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Risk Level Proportions')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Risk analysis saved to {save_path}")
    
    return risk_counts


def main():
    """Main testing function."""
    print("\n" + "="*70)
    print(" " * 15 + "GAMER ERGOVISION - MODEL TESTING")
    print("="*70 + "\n")
    
    # Check if model exists
    if not POSTURE_CLASSIFIER_PATH.exists():
        print(f"Error: No trained model found at {POSTURE_CLASSIFIER_PATH}")
        print("Please train a model first using train_classifier.py")
        return
    
    # Create results directory
    results_dir = RESULTS_ROOT / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = torch.device(CLASSIFIER_CONFIG["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load test dataset
    print("Loading test dataset...")
    test_transform = get_posture_transforms("test")
    test_dataset = PostureDataset(YOLO_DATASET_PATH, split="test", transform=test_transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CLASSIFIER_CONFIG["batch_size"],
        shuffle=False,
        num_workers=CLASSIFIER_CONFIG["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Load model
    print(f"Loading model from {POSTURE_CLASSIFIER_PATH}...")
    model = PostureClassifier(
        num_classes=len(POSTURE_CLASSES),
        backbone=CLASSIFIER_CONFIG["backbone"],
        pretrained=False
    )
    
    checkpoint = torch.load(POSTURE_CLASSIFIER_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Best validation accuracy: {checkpoint.get('val_acc', 0):.2f}%\n")
    
    # Run testing
    print("="*70)
    print("Running Test Evaluation")
    print("="*70 + "\n")
    
    predictions, labels, confidences = test_model(model, test_loader, device)
    
    # Calculate metrics
    test_acc = accuracy_score(labels, predictions) * 100
    print(f"\nTest Accuracy: {test_acc:.2f}%\n")
    
    # Display names for reports
    display_labels = [POSTURE_DISPLAY_NAMES.get(cls, cls) for cls in POSTURE_CLASSES]
    
    # Classification report
    print("="*70)
    print("Detailed Classification Report")
    print("="*70)
    report = classification_report(labels, predictions, target_names=display_labels)
    print(report)
    
    # Save report to file
    report_path = results_dir / "test_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("GAMER ERGOVISION - TEST SET CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Total Test Samples: {len(labels)}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n" + "="*70 + "\n")
        f.write("Per-Class Sample Counts:\n")
        for i, class_name in enumerate(POSTURE_CLASSES):
            count = (labels == i).sum()
            display_name = POSTURE_DISPLAY_NAMES.get(class_name, class_name)
            f.write(f"  {display_name}: {count}\n")
    
    print(f"✓ Report saved to {report_path}\n")
    
    # Generate visualizations
    print("="*70)
    print("Generating Visualizations")
    print("="*70 + "\n")
    
    # Confusion matrix
    plot_confusion_matrix(
        labels, predictions, POSTURE_CLASSES,
        results_dir / "test_confusion_matrix.png"
    )
    
    # Confidence distribution
    plot_confidence_distribution(
        confidences, predictions, labels,
        results_dir / "confidence_distribution.png"
    )
    
    # Per-class analysis
    analyze_per_class_performance(
        labels, predictions, confidences, POSTURE_CLASSES,
        results_dir / "per_class_analysis.png"
    )
    
    # Risk analysis
    print("\n" + "="*70)
    print("Ergonomic Risk Analysis")
    print("="*70)
    risk_counts = generate_risk_analysis(
        predictions, POSTURE_CLASSES,
        results_dir / "risk_distribution.png"
    )
    
    print("\nPredicted Risk Distribution:")
    total = sum(risk_counts.values())
    for risk, count in risk_counts.items():
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {risk.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"Total test samples: {len(labels)}")
    print(f"Correct predictions: {(predictions == labels).sum()}")
    print(f"Incorrect predictions: {(predictions != labels).sum()}")
    print(f"Overall accuracy: {test_acc:.2f}%")
    print(f"Average confidence: {confidences.mean():.3f}")
    print(f"Confidence std dev: {confidences.std():.3f}")
    
    print("\n" + "="*70)
    print("Testing Complete!")
    print(f"All results saved to: {results_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
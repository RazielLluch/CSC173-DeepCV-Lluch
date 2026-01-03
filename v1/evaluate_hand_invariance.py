"""
evaluate_hand_invariance.py
Comprehensive evaluation of model's sensitivity to hand position.
Compares baseline vs invariant model performance.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

from config import POSTURE_CLASSES, YOLO_DATASET_PATH, RESULTS_ROOT
from pose_features import PoseFeatureExtractor
from datasets import get_posture_transforms
from PIL import Image as PILImage


class HandInvarianceEvaluator:
    """
    Evaluate model's invariance to hand position.
    
    Strategy:
    1. Create synthetic test cases: shift hand regions vertically
    2. Measure prediction consistency
    3. Compare baseline vs invariant model
    """
    
    def __init__(self, baseline_model_path=None, invariant_model_path=None):
        """
        Initialize evaluator with both models.
        
        Args:
            baseline_model_path: Path to original model
            invariant_model_path: Path to invariant model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load baseline model
        if baseline_model_path:
            print(f"Loading baseline model from {baseline_model_path}")
            from train_classifier import PostureClassifier
            self.baseline_model = PostureClassifier(
                num_classes=len(POSTURE_CLASSES),
                backbone="resnet18"
            )
            checkpoint = torch.load(baseline_model_path, map_location=self.device)
            self.baseline_model.load_state_dict(checkpoint['model_state_dict'])
            self.baseline_model.to(self.device)
            self.baseline_model.eval()
        else:
            self.baseline_model = None
            print("No baseline model provided")
        
        # Load invariant model
        if invariant_model_path:
            print(f"Loading invariant model from {invariant_model_path}")
            from hybrid_model import HybridPostureClassifier
            self.invariant_model = HybridPostureClassifier(
                num_classes=len(POSTURE_CLASSES),
                pose_feature_dim=24
            )
            checkpoint = torch.load(invariant_model_path, map_location=self.device)
            self.invariant_model.load_state_dict(checkpoint['model_state_dict'])
            self.invariant_model.to(self.device)
            self.invariant_model.eval()
            
            # Pose extractor
            from pose_features import SideViewPoseFeatureExtractor
            self.pose_extractor = SideViewPoseFeatureExtractor()
        else:
            self.invariant_model = None
            self.pose_extractor = None
            print("No invariant model provided")
        
        # Transform
        self.transform = get_posture_transforms("test")
    
    def create_hand_shifted_versions(self, image, num_shifts=5):
        """
        Create versions of image with hands in different vertical positions.
        
        Args:
            image: Original image
            num_shifts: Number of shifted versions to create
        
        Returns:
            List of (shifted_image, shift_amount) tuples
        """
        h, w = image.shape[:2]
        shifted_versions = [(image, 0)]  # Original
        
        # Create shifts from -20% to +20% of image height
        for shift_factor in np.linspace(-0.2, 0.2, num_shifts):
            if shift_factor == 0:
                continue
            
            shift_pixels = int(h * shift_factor)
            M = np.float32([[1, 0, 0], [0, 1, shift_pixels]])
            shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            shifted_versions.append((shifted, shift_pixels))
        
        return shifted_versions
    
    def predict_baseline(self, image):
        """Predict using baseline model."""
        if self.baseline_model is None:
            return None
        
        # Transform
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.baseline_model(image_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()[0]
    
    def predict_invariant(self, image):
        """Predict using invariant model."""
        if self.invariant_model is None or self.pose_extractor is None:
            return None
        
        # Extract pose features
        pose_result = self.pose_extractor.extract_features(image)
        if pose_result is None:
            return None
        
        features, _ = pose_result
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # Transform image
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.invariant_model(image_tensor, features_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()[0]
    
    def compute_prediction_stability(self, probabilities_list):
        """
        Measure how stable predictions are across shifted versions.
        
        Args:
            probabilities_list: List of probability vectors
        
        Returns:
            Dictionary with stability metrics
        """
        if not probabilities_list:
            return None
        
        probs_array = np.array(probabilities_list)  # [num_shifts, num_classes]
        
        # 1. Prediction consistency: do we predict same class?
        predicted_classes = np.argmax(probs_array, axis=1)
        most_common = np.bincount(predicted_classes).argmax()
        consistency = (predicted_classes == most_common).mean()
        
        # 2. Confidence stability: std of max probabilities
        max_probs = np.max(probs_array, axis=1)
        confidence_std = np.std(max_probs)
        confidence_mean = np.mean(max_probs)
        
        # 3. Probability vector stability: average KL divergence from mean
        mean_probs = np.mean(probs_array, axis=0)
        kl_divs = []
        for probs in probs_array:
            # KL(probs || mean_probs)
            kl = np.sum(probs * np.log((probs + 1e-10) / (mean_probs + 1e-10)))
            kl_divs.append(kl)
        avg_kl_divergence = np.mean(kl_divs)
        
        return {
            'consistency': consistency,
            'confidence_mean': confidence_mean,
            'confidence_std': confidence_std,
            'kl_divergence': avg_kl_divergence,
            'predicted_class': most_common
        }
    
    def evaluate_on_dataset(self, data_path, split="test", max_samples=None):
        """
        Evaluate on entire dataset.
        
        Args:
            data_path: Path to dataset
            split: train/val/test
            max_samples: Maximum samples to evaluate (None for all)
        
        Returns:
            DataFrame with results
        """
        from datasets import PostureDataset
        
        # Load dataset
        dataset = PostureDataset(data_path, split=split, transform=None)
        
        if max_samples:
            samples = dataset.samples[:max_samples]
        else:
            samples = dataset.samples
        
        results = []
        
        print(f"\nEvaluating on {len(samples)} samples from {split} set...")
        
        for img_path, class_name in tqdm(samples):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Create shifted versions
            shifted_versions = self.create_hand_shifted_versions(image, num_shifts=5)
            
            # Baseline predictions
            baseline_probs = []
            if self.baseline_model is not None:
                for shifted_img, shift in shifted_versions:
                    probs = self.predict_baseline(shifted_img)
                    if probs is not None:
                        baseline_probs.append(probs)
            
            # Invariant predictions
            invariant_probs = []
            if self.invariant_model is not None:
                for shifted_img, shift in shifted_versions:
                    probs = self.predict_invariant(shifted_img)
                    if probs is not None:
                        invariant_probs.append(probs)
            
            # Compute stability
            baseline_stability = self.compute_prediction_stability(baseline_probs)
            invariant_stability = self.compute_prediction_stability(invariant_probs)
            
            # Store results
            result = {
                'image': img_path.name,
                'true_class': class_name,
                'true_class_idx': POSTURE_CLASSES.index(class_name)
            }
            
            if baseline_stability:
                result.update({
                    'baseline_consistency': baseline_stability['consistency'],
                    'baseline_confidence_mean': baseline_stability['confidence_mean'],
                    'baseline_confidence_std': baseline_stability['confidence_std'],
                    'baseline_kl_div': baseline_stability['kl_divergence'],
                    'baseline_pred': baseline_stability['predicted_class'],
                    'baseline_correct': baseline_stability['predicted_class'] == result['true_class_idx']
                })
            
            if invariant_stability:
                result.update({
                    'invariant_consistency': invariant_stability['consistency'],
                    'invariant_confidence_mean': invariant_stability['confidence_mean'],
                    'invariant_confidence_std': invariant_stability['confidence_std'],
                    'invariant_kl_div': invariant_stability['kl_divergence'],
                    'invariant_pred': invariant_stability['predicted_class'],
                    'invariant_correct': invariant_stability['predicted_class'] == result['true_class_idx']
                })
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, results_df, save_path):
        """
        Create comparison plots.
        
        Args:
            results_df: DataFrame from evaluate_on_dataset
            save_path: Where to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Prediction consistency comparison
        ax = axes[0, 0]
        if 'baseline_consistency' in results_df.columns:
            baseline_mean = results_df['baseline_consistency'].mean()
            ax.bar(0, baseline_mean, label='Baseline', color='coral', alpha=0.7)
        if 'invariant_consistency' in results_df.columns:
            invariant_mean = results_df['invariant_consistency'].mean()
            ax.bar(1, invariant_mean, label='Invariant', color='skyblue', alpha=0.7)
        ax.set_ylabel('Consistency (higher = better)')
        ax.set_title('Prediction Consistency Across Hand Positions')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Baseline', 'Invariant'])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # 2. Confidence stability
        ax = axes[0, 1]
        data_to_plot = []
        labels = []
        if 'baseline_confidence_std' in results_df.columns:
            data_to_plot.append(results_df['baseline_confidence_std'].dropna())
            labels.append('Baseline')
        if 'invariant_confidence_std' in results_df.columns:
            data_to_plot.append(results_df['invariant_confidence_std'].dropna())
            labels.append('Invariant')
        
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_ylabel('Confidence Std Dev (lower = better)')
            ax.set_title('Confidence Stability')
            ax.grid(True, alpha=0.3)
        
        # 3. KL divergence (probability stability)
        ax = axes[1, 0]
        data_to_plot = []
        labels = []
        if 'baseline_kl_div' in results_df.columns:
            data_to_plot.append(results_df['baseline_kl_div'].dropna())
            labels.append('Baseline')
        if 'invariant_kl_div' in results_df.columns:
            data_to_plot.append(results_df['invariant_kl_div'].dropna())
            labels.append('Invariant')
        
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_ylabel('KL Divergence (lower = better)')
            ax.set_title('Probability Distribution Stability')
            ax.grid(True, alpha=0.3)
        
        # 4. Accuracy comparison
        ax = axes[1, 1]
        accuracies = []
        labels = []
        if 'baseline_correct' in results_df.columns:
            baseline_acc = results_df['baseline_correct'].mean()
            accuracies.append(baseline_acc)
            labels.append(f'Baseline\n({baseline_acc:.1%})')
        if 'invariant_correct' in results_df.columns:
            invariant_acc = results_df['invariant_correct'].mean()
            accuracies.append(invariant_acc)
            labels.append(f'Invariant\n({invariant_acc:.1%})')
        
        if accuracies:
            ax.bar(range(len(accuracies)), accuracies, 
                   color=['coral', 'skyblue'][:len(accuracies)], alpha=0.7)
            ax.set_ylabel('Accuracy')
            ax.set_title('Overall Accuracy')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison plots saved to {save_path}")
    
    def generate_report(self, results_df, save_path):
        """Generate text report."""
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HAND-INVARIANCE EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("Metrics (higher consistency = better invariance):\n")
            f.write("-"*70 + "\n\n")
            
            if 'baseline_consistency' in results_df.columns:
                f.write("BASELINE MODEL:\n")
                f.write(f"  Prediction Consistency: {results_df['baseline_consistency'].mean():.3f} ± {results_df['baseline_consistency'].std():.3f}\n")
                f.write(f"  Confidence Stability: {results_df['baseline_confidence_std'].mean():.4f} (std dev)\n")
                f.write(f"  Probability Stability: {results_df['baseline_kl_div'].mean():.4f} (KL div)\n")
                f.write(f"  Accuracy: {results_df['baseline_correct'].mean():.3f}\n\n")
            
            if 'invariant_consistency' in results_df.columns:
                f.write("INVARIANT MODEL:\n")
                f.write(f"  Prediction Consistency: {results_df['invariant_consistency'].mean():.3f} ± {results_df['invariant_consistency'].std():.3f}\n")
                f.write(f"  Confidence Stability: {results_df['invariant_confidence_std'].mean():.4f} (std dev)\n")
                f.write(f"  Probability Stability: {results_df['invariant_kl_div'].mean():.4f} (KL div)\n")
                f.write(f"  Accuracy: {results_df['invariant_correct'].mean():.3f}\n\n")
            
            if 'baseline_consistency' in results_df.columns and 'invariant_consistency' in results_df.columns:
                improvement = (results_df['invariant_consistency'].mean() - 
                              results_df['baseline_consistency'].mean()) * 100
                f.write(f"IMPROVEMENT:\n")
                f.write(f"  Consistency gain: {improvement:+.1f} percentage points\n")
                
                conf_improvement = (results_df['baseline_confidence_std'].mean() - 
                                   results_df['invariant_confidence_std'].mean())
                f.write(f"  Confidence stability gain: {conf_improvement:+.4f} (lower is better)\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✓ Report saved to {save_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate hand-position invariance")
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline model')
    parser.add_argument('--invariant', type=str, default=None,
                       help='Path to invariant model')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to evaluate')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" " * 15 + "HAND-INVARIANCE EVALUATION")
    print("="*70 + "\n")
    
    # Initialize evaluator
    evaluator = HandInvarianceEvaluator(
        baseline_model_path=args.baseline,
        invariant_model_path=args.invariant
    )
    
    # Run evaluation
    results_df = evaluator.evaluate_on_dataset(
        YOLO_DATASET_PATH,
        split=args.split,
        max_samples=args.max_samples
    )
    
    # Create output directory
    output_dir = RESULTS_ROOT / "hand_invariance_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_csv = output_dir / "detailed_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Detailed results saved to {results_csv}")
    
    # Generate plots
    plot_path = output_dir / "comparison_plots.png"
    evaluator.plot_comparison(results_df, plot_path)
    
    # Generate report
    report_path = output_dir / "evaluation_report.txt"
    evaluator.generate_report(results_df, report_path)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if 'baseline_consistency' in results_df.columns:
        print(f"\nBaseline Model:")
        print(f"  Consistency: {results_df['baseline_consistency'].mean():.1%}")
        print(f"  Accuracy: {results_df['baseline_correct'].mean():.1%}")
    
    if 'invariant_consistency' in results_df.columns:
        print(f"\nInvariant Model:")
        print(f"  Consistency: {results_df['invariant_consistency'].mean():.1%}")
        print(f"  Accuracy: {results_df['invariant_correct'].mean():.1%}")
    
    if 'baseline_consistency' in results_df.columns and 'invariant_consistency' in results_df.columns:
        improvement = (results_df['invariant_consistency'].mean() - 
                      results_df['baseline_consistency'].mean()) * 100
        print(f"\nImprovement: {improvement:+.1f} percentage points in consistency")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
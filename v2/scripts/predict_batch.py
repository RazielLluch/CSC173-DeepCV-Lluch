"""
Batch prediction script for processing multiple images.
Generates predictions for all images in a directory and creates statistics.
"""

import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import argparse

from config import POSTURE_CLASSES, POSTURE_RISK_MAPPING, get_device, get_model_save_path
from scripts.predict_single import PosturePredictor


class BatchPredictor:
    """
    Batch prediction for multiple images.
    """
    
    def __init__(self, checkpoint_path: Path = None):
        """
        Initialize batch predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        self.predictor = PosturePredictor(checkpoint_path=checkpoint_path)
        self.results = []
    
    def predict_directory(
        self,
        input_dir: str,
        extensions: List[str] = ['.jpg', '.jpeg', '.png'],
        recursive: bool = False
    ) -> List[Dict]:
        """
        Predict posture for all images in a directory.
        
        Args:
            input_dir: Directory containing images
            extensions: List of valid image extensions
            recursive: Whether to search subdirectories
            
        Returns:
            List of prediction dictionaries
        """
        input_path = Path(input_dir)
        
        # Collect all image files
        image_files = []
        for ext in extensions:
            if recursive:
                image_files.extend(list(input_path.rglob(f'*{ext}')))
            else:
                image_files.extend(list(input_path.glob(f'*{ext}')))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        if len(image_files) == 0:
            print("No images found!")
            return []
        
        # Process each image
        self.results = []
        failed_count = 0
        
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                result = self.predictor.predict(str(image_path))
                
                if result is not None:
                    # Add filename to results
                    result['filename'] = image_path.name
                    result['filepath'] = str(image_path)
                    self.results.append(result)
                else:
                    failed_count += 1
                    print(f"\nFailed to detect pose in: {image_path.name}")
                    
            except Exception as e:
                failed_count += 1
                print(f"\nError processing {image_path.name}: {e}")
        
        print(f"\nSuccessfully processed: {len(self.results)}/{len(image_files)}")
        print(f"Failed: {failed_count}")
        
        return self.results
    
    def generate_statistics(self) -> pd.DataFrame:
        """
        Generate statistics from predictions.
        
        Returns:
            DataFrame with prediction statistics
        """
        if not self.results:
            print("No results to analyze!")
            return None
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'filename': r['filename'],
                'class': r['class'],
                'confidence': r['confidence'],
                'risk_level': r['risk_level'],
                'spine_angle': r['spine_angle'],
                **{f'prob_{cls}': r['probabilities'][cls] for cls in POSTURE_CLASSES}
            }
            for r in self.results
        ])
        
        return df
    
    def plot_statistics(self, save_dir: Path):
        """
        Create visualization of batch predictions.
        
        Args:
            save_dir: Directory to save plots
        """
        if not self.results:
            print("No results to plot!")
            return
        
        df = self.generate_statistics()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Class distribution
        class_counts = df['class'].value_counts()
        axes[0, 0].bar(class_counts.index, class_counts.values, 
                      color=['green', 'orange', 'red'], alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Posture Class', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add count labels
        for i, (idx, count) in enumerate(class_counts.items()):
            axes[0, 0].text(i, count + 0.5, str(count), ha='center', fontsize=11)
        
        # 2. Confidence distribution by class
        for cls in POSTURE_CLASSES:
            cls_df = df[df['class'] == cls]
            if len(cls_df) > 0:
                axes[0, 1].hist(cls_df['confidence'], bins=20, alpha=0.5, label=cls)
        axes[0, 1].set_xlabel('Confidence', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Confidence Distribution by Class', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Risk level distribution
        risk_counts = df['risk_level'].value_counts()
        colors_risk = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
        axes[1, 0].bar(
            risk_counts.index,
            risk_counts.values,
            color=[colors_risk.get(r, 'gray') for r in risk_counts.index],
            alpha=0.7,
            edgecolor='black'
        )
        axes[1, 0].set_xlabel('Risk Level', fontsize=12)
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add percentage labels
        total = len(df)
        for i, (idx, count) in enumerate(risk_counts.items()):
            percentage = (count / total) * 100
            axes[1, 0].text(i, count + 0.5, f'{count}\n({percentage:.1f}%)', 
                          ha='center', fontsize=10)
        
        # 4. Spine angle distribution
        for cls in POSTURE_CLASSES:
            cls_df = df[df['class'] == cls]
            if len(cls_df) > 0:
                axes[1, 1].hist(cls_df['spine_angle'], bins=20, alpha=0.5, label=cls)
        axes[1, 1].set_xlabel('Spine Angle (degrees)', fontsize=12)
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Spine Angle Distribution by Class', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = save_dir / 'batch_statistics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot saved to {plot_path}")
        plt.close()
    
    def save_results(self, save_dir: Path, csv_filename: str = 'predictions.csv'):
        """
        Save prediction results to CSV.
        
        Args:
            save_dir: Directory to save results
            csv_filename: Name of CSV file
        """
        if not self.results:
            print("No results to save!")
            return
        
        df = self.generate_statistics()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = save_dir / csv_filename
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("BATCH PREDICTION SUMMARY")
        print("="*80)
        print(f"Total Images Processed: {len(df)}")
        print("\nClass Distribution:")
        print(df['class'].value_counts())
        print("\nRisk Level Distribution:")
        print(df['risk_level'].value_counts())
        print("\nAverage Confidence by Class:")
        print(df.groupby('class')['confidence'].mean())
        print("\nAverage Spine Angle by Class:")
        print(df.groupby('class')['spine_angle'].mean())
        print("="*80)


def main():
    """
    Main batch prediction function.
    """
    parser = argparse.ArgumentParser(description='Batch posture prediction')
    parser.add_argument('input_dir', type=str, help='Directory containing images')
    parser.add_argument('--output-dir', type=str, default='results/batch',
                       help='Directory to save results')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories recursively')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GAMER ERGOVISION - BATCH PREDICTION")
    print("="*80)
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Recursive: {args.recursive}")
    print("="*80 + "\n")
    
    # Create predictor
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    predictor = BatchPredictor(checkpoint_path=checkpoint)
    
    # Process directory
    results = predictor.predict_directory(
        args.input_dir,
        recursive=args.recursive
    )
    
    if results:
        # Save results
        output_dir = Path(args.output_dir)
        predictor.save_results(output_dir)
        
        # Generate plots
        predictor.plot_statistics(output_dir)
        
        print(f"\nAll results saved to {output_dir}")
    else:
        print("\nNo successful predictions to save!")


if __name__ == "__main__":
    main()
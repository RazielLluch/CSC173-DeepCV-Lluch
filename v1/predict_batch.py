"""
predict_batch.py
Predict posture for multiple images in a folder.
"""

import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from collections import defaultdict

from config import (
    POSTURE_CLASSES, POSTURE_DISPLAY_NAMES, POSTURE_RISK_MAPPING,
    RESULTS_ROOT
)
from predict_single import PosturePredictor, print_prediction_results


def predict_folder(
    folder_path: str,
    output_dir: str = None,
    save_visualizations: bool = True,
    save_json: bool = True,
    file_extensions: list = None,
    model_path: str = Path(r"C:/Users/Admin/Desktop/School/CSC173/final project/CSC173-DeepCV-Lluch/results/invariant_training/best_model_hybrid.pth")
):
    """
    Predict posture for all images in a folder.
    
    Args:
        folder_path: Path to folder containing images
        output_dir: Directory to save results (default: results/batch_predictions)
        save_visualizations: Whether to save visualization images
        save_json: Whether to save JSON results
        file_extensions: List of file extensions to process
    
    Returns:
        List of prediction results
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Default output directory
    if output_dir is None:
        output_dir = RESULTS_ROOT / "batch_predictions" / folder_path.name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default file extensions
    if file_extensions is None:
        file_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Find all images
    image_files = []
    for ext in file_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return []
    
    print(f"\nFound {len(image_files)} images in {folder_path}")
    print(f"Output directory: {output_dir}\n")
    
    # Initialize predictor
    predictor = PosturePredictor(model_path=model_path)
    
    # Process images
    all_results = []
    stats = defaultdict(int)
    
    print("Processing images...")
    for img_path in tqdm(image_files, desc="Predicting"):
        try:
            # Run prediction
            results = predictor.predict(str(img_path))
            all_results.append(results)
            
            # Update statistics
            stats[results['predicted_class']] += 1
            stats[f"risk_{results['risk_level']}"] += 1
            
            # Save visualization if requested
            if save_visualizations:
                save_path = output_dir / "visualizations" / f"{img_path.stem}_prediction.jpg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create visualization
                _, _ = predictor.predict_and_visualize(
                    str(img_path),
                    save_path=str(save_path),
                    show=False
                )
        
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue
    
    # Save JSON results
    if save_json:
        json_path = output_dir / "predictions.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ JSON results saved to {json_path}")
    
    # Print summary
    print("\n" + "="*70)
    print(" " * 20 + "BATCH PREDICTION SUMMARY")
    print("="*70)
    print(f"\nTotal images processed: {len(all_results)}")
    
    print("\nPosture Distribution:")
    print("-"*70)
    for class_name in POSTURE_CLASSES:
        count = stats[class_name]
        percentage = (count / len(all_results) * 100) if all_results else 0
        display_name = POSTURE_DISPLAY_NAMES.get(class_name, class_name)
        print(f"  {display_name:<25} {count:>5} ({percentage:>5.1f}%)")
    
    print("\nRisk Level Distribution:")
    print("-"*70)
    for risk in ['low', 'medium', 'high']:
        count = stats[f"risk_{risk}"]
        percentage = (count / len(all_results) * 100) if all_results else 0
        print(f"  {risk.capitalize():<25} {count:>5} ({percentage:>5.1f}%)")
    
    print("\n" + "="*70)
    
    # Save summary to text file
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("GAMER ERGOVISION - BATCH PREDICTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total images processed: {len(all_results)}\n\n")
        
        f.write("Posture Distribution:\n")
        f.write("-"*70 + "\n")
        for class_name in POSTURE_CLASSES:
            count = stats[class_name]
            percentage = (count / len(all_results) * 100) if all_results else 0
            display_name = POSTURE_DISPLAY_NAMES.get(class_name, class_name)
            f.write(f"  {display_name:<25} {count:>5} ({percentage:>5.1f}%)\n")
        
        f.write("\nRisk Level Distribution:\n")
        f.write("-"*70 + "\n")
        for risk in ['low', 'medium', 'high']:
            count = stats[f"risk_{risk}"]
            percentage = (count / len(all_results) * 100) if all_results else 0
            f.write(f"  {risk.capitalize():<25} {count:>5} ({percentage:>5.1f}%)\n")
    
    print(f"\n✓ Summary saved to {summary_path}")
    print(f"\nAll results saved to: {output_dir}\n")
    
    return all_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Predict posture for multiple images in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a folder
  python predict_batch.py path/to/folder
  
  # Process and specify output directory
  python predict_batch.py path/to/folder --output results/my_predictions
  
  # Just predict, don't save visualizations
  python predict_batch.py path/to/folder --no-visualizations
  
  # Don't save JSON file
  python predict_batch.py path/to/folder --no-json
        """
    )
    
    parser.add_argument('folder_path', type=str,
                       help='Path to folder containing images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Do not save visualization images')
    parser.add_argument('--no-json', action='store_true',
                       help='Do not save JSON results')
    parser.add_argument('--extensions', nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'],
                       help='File extensions to process')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights (default: use config path)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print(" " * 15 + "GAMER ERGOVISION - BATCH PREDICTION")
    print("="*70)
    
    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Run batch prediction
    try:
        results = predict_folder(
            args.folder_path,
            output_dir=args.output,
            save_visualizations=not args.no_visualizations,
            save_json=not args.no_json,
            file_extensions=args.extensions
        )
        
        if not results:
            print("❌ No images were successfully processed")
            return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
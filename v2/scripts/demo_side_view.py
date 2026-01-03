"""
Simple demo script for visualizing pose detection and feature extraction.
Useful for testing MediaPipe setup and understanding the pose features.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from utils.pose_features import SideViewPoseFeatureExtractor


def demo_pose_extraction(image_path: str, save_path: str = None, show: bool = True):
    """
    Demonstrate pose extraction on a single image.
    
    Args:
        image_path: Path to input image
        save_path: Optional path to save visualization
        show: Whether to display the plot
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Initialize pose extractor
    print("Extracting pose features...")
    extractor = SideViewPoseFeatureExtractor()
    
    # Extract features
    result = extractor.extract_features(image)
    
    if result is None:
        print("Error: Could not detect pose in image")
        return
    
    features, debug_info = result
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Pose overlay
    pose_image = extractor.visualize_pose(
        image,
        debug_info['landmarks'],
        debug_info['side']
    )
    pose_image_rgb = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
    axes[1].imshow(pose_image_rgb)
    axes[1].set_title(f'Detected Pose (Side: {debug_info["side"]})')
    axes[1].axis('off')
    
    plt.suptitle('Side-View Pose Detection Demo', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print feature information
    print("\n" + "="*80)
    print("EXTRACTED POSE FEATURES")
    print("="*80)
    print(f"Detected Side: {debug_info['side']}")
    print(f"Spine Angle: {debug_info['spine_angle']:.2f}°")
    print(f"Head Forward Distance: {debug_info['head_forward_distance']:.3f}")
    print(f"Shoulder Rotation: {debug_info['shoulder_rotation']:.3f}")
    print(f"\nTotal Features Extracted: {len(features)}")
    print("\nFeature Vector (first 10 values):")
    print(features[:10])
    print("="*80)


def demo_feature_visualization(image_path: str, save_path: str = None, show: bool = True):
    """
    Create a detailed visualization of all 24 pose features.
    
    Args:
        image_path: Path to input image
        save_path: Optional path to save visualization
        show: Whether to display the plot
    """
    # Load image and extract features
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    extractor = SideViewPoseFeatureExtractor()
    result = extractor.extract_features(image)
    
    if result is None:
        print("Error: Could not detect pose in image")
        return
    
    features, debug_info = result
    
    # Feature names (24 total)
    feature_names = [
        'Head Forward', 'Head Vertical', 'Head Tilt',
        'Spine Angle Norm', 'Spine Angle Raw', 'Upper Curvature',
        'Lower Curvature', 'Spine Straightness', 'Shoulder Height Diff',
        'Shoulder Rotation', 'Shoulder Horizontal', 'Shoulder Alignment',
        'Torso Aspect', 'Torso Width', 'Torso Height',
        'Torso Compactness', 'COM X', 'COM Y',
        'Elbow Angle L', 'Elbow Angle R', 'Nose X', 'Nose Y',
        'Shoulder X', 'Shoulder Y'
    ]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Original + Pose overlay (top row)
    ax1 = fig.add_subplot(gs[0, :2])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 2])
    pose_image = extractor.visualize_pose(image, debug_info['landmarks'], debug_info['side'])
    pose_image_rgb = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
    ax2.imshow(pose_image_rgb)
    ax2.set_title('Pose Landmarks', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Feature bar chart (middle row)
    ax3 = fig.add_subplot(gs[1, :])
    colors = ['#FF6B6B' if f > 0.5 else '#4ECDC4' for f in features]
    bars = ax3.bar(range(len(features)), features, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Feature Index', fontsize=12)
    ax3.set_ylabel('Feature Value', fontsize=12)
    ax3.set_title('24 Geometric Pose Features', fontsize=14, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(len(features)))
    ax3.set_xticklabels(range(1, len(features)+1), fontsize=8)
    
    # Key metrics (bottom row)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    metrics_text = "KEY METRICS\n\n"
    metrics_text += f"Detected Side: {debug_info['side']}\n\n"
    metrics_text += "CRITICAL FEATURES:\n"
    metrics_text += f"  • Spine Angle: {debug_info['spine_angle']:.2f}°\n"
    metrics_text += f"  • Head Forward: {features[0]:.3f}\n"
    metrics_text += f"  • Spine Straightness: {features[7]:.3f}\n"
    metrics_text += f"  • Torso Aspect Ratio: {features[12]:.3f}\n\n"
    
    metrics_text += "FEATURE CATEGORIES:\n"
    metrics_text += "  • Head Position: features 0-2\n"
    metrics_text += "  • Spine Alignment: features 3-7\n"
    metrics_text += "  • Shoulder Position: features 8-11\n"
    metrics_text += "  • Upper Body Geometry: features 12-17\n"
    metrics_text += "  • Elbow Angles: features 18-19\n"
    metrics_text += "  • Normalized Coordinates: features 20-23"
    
    ax4.text(
        0.5, 0.5,
        metrics_text,
        fontsize=11,
        ha='center',
        va='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    plt.suptitle('Detailed Pose Feature Analysis', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Detailed visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description='Demo script for pose extraction and visualization'
    )
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--mode', type=str, default='simple',
                       choices=['simple', 'detailed'],
                       help='Visualization mode: simple or detailed')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display visualization')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GAMER ERGOVISION - POSE EXTRACTION DEMO")
    print("="*80)
    print(f"Input Image: {args.image_path}")
    print(f"Mode: {args.mode}")
    print("="*80 + "\n")
    
    if args.mode == 'simple':
        demo_pose_extraction(
            args.image_path,
            save_path=args.output,
            show=not args.no_show
        )
    else:
        demo_feature_visualization(
            args.image_path,
            save_path=args.output,
            show=not args.no_show
        )


if __name__ == "__main__":
    # If run without arguments, show help
    import sys
    if len(sys.argv) == 1:
        print("="*80)
        print("GAMER ERGOVISION - POSE EXTRACTION DEMO")
        print("="*80)
        print("\nUsage:")
        print("  python demo_side_view.py <image_path> [options]")
        print("\nOptions:")
        print("  --mode {simple,detailed}  Visualization mode (default: simple)")
        print("  --output PATH             Save visualization to file")
        print("  --no-show                 Don't display visualization")
        print("\nExamples:")
        print("  python demo_side_view.py image.jpg")
        print("  python demo_side_view.py image.jpg --mode detailed --output result.png")
        print("="*80)
    else:
        main()
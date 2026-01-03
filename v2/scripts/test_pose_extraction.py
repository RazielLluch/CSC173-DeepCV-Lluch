"""
Test script to verify pose extraction works correctly.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from ..utils.pose_features import SideViewPoseFeatureExtractor

def test_pose_extraction():
    """Test pose extraction with a simple generated image."""
    
    print("="*60)
    print("TESTING POSE EXTRACTION")
    print("="*60)
    
    # Create a simple test image (just for testing - won't detect pose)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Initialize extractor
    print("\n1. Initializing pose extractor...")
    extractor = SideViewPoseFeatureExtractor()
    print("   ✅ Extractor initialized")
    
    # Try extracting features
    print("\n2. Attempting feature extraction on blank image...")
    result = extractor.extract_features(test_image)
    
    if result is None:
        print("   ⚠️  No pose detected (expected for blank image)")
        print("   This is normal - please test with an actual person image")
    else:
        features, debug_info = result
        print("   ✅ Features extracted successfully!")
        print(f"   Features shape: {features.shape}")
        print(f"   Spine angle: {debug_info['spine_angle']:.2f}°")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\n💡 Next step: Test with an actual image of a person")
    print("   Usage: python test_pose_extraction.py path/to/image.jpg")
    
    return extractor

def test_with_image(image_path):
    """Test with an actual image file."""
    
    print(f"\nTesting with image: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Initialize extractor
    extractor = SideViewPoseFeatureExtractor()
    
    # Extract features
    print("Extracting features...")
    result = extractor.extract_features(image)
    
    if result is None:
        print("❌ Could not detect pose in image")
        print("   Make sure the image shows a person in side view")
        return
    
    features, debug_info = result
    
    print("\n" + "="*60)
    print("POSE EXTRACTION SUCCESS")
    print("="*60)
    print(f"Detected Side: {debug_info['side']}")
    print(f"Spine Angle: {debug_info['spine_angle']:.2f}°")
    print(f"Head Forward Distance: {debug_info['head_forward_distance']:.3f}")
    print(f"Total Features: {len(features)}")
    print("="*60)
    
    # Try visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # With pose
        pose_image = extractor.visualize_pose(image, debug_info['landmarks'], debug_info['side'])
        pose_rgb = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
        axes[1].imshow(pose_rgb)
        axes[1].set_title(f"Detected Pose (Side: {debug_info['side']})")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('pose_test_result.png', dpi=150, bbox_inches='tight')
        print("\n✅ Visualization saved to: pose_test_result.png")
        plt.show()
        
    except Exception as e:
        print(f"\n⚠️  Visualization failed: {e}")
        print("   But feature extraction worked!")

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         # Test with provided image
#         test_with_image(sys.argv[1])
#     else:
#         # Basic test
#         test_pose_extraction()
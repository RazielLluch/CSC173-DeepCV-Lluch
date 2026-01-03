# simple_demo.py
import cv2
import torch
import numpy as np
from pose_features import SideViewPoseFeatureExtractor
from hybrid_model import HybridPostureClassifier
from datasets import get_posture_transforms
from config import POSTURE_CLASSES, POSTURE_DISPLAY_NAMES, POSTURE_RISK_MAPPING
from PIL import Image as PILImage
from pathlib import Path

def simple_prediction(image_path):
    """Simple prediction on one image."""
    print("\n" + "="*70)
    print("SIMPLE POSTURE PREDICTION")
    print("="*70 + "\n")
    
    # Check file exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Could not load image")
        return
    
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Extract pose features
    print("\nExtracting pose features...")
    extractor = SideViewPoseFeatureExtractor()
    
    result = extractor.extract_features(image)
    
    if result is None:
        print("❌ Could not extract pose from image")
        print("   Make sure image shows person in side-view")
        return
    
    features, debug_info = result
    print("✓ Pose extracted successfully!")
    print(f"  Visible side: {debug_info['visible_side']}")
    print(f"  Spine angle: {np.degrees(debug_info['spine_angle']):.1f}°")
    print(f"  Head forward: {debug_info['head_forward']:.3f}")
    
    # Load model
    print("\nLoading hybrid model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridPostureClassifier(num_classes=3, pose_feature_dim=24)
    
    checkpoint = torch.load("results/invariant_training/best_model_hybrid.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Prepare input
    print("\nMaking prediction...")
    transform = get_posture_transforms("test")
    
    pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    feat_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor, feat_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    pred_class = POSTURE_CLASSES[predicted.item()]
    conf = confidence.item()
    
    # Get risk level
    risk = POSTURE_RISK_MAPPING.get(pred_class, "medium")
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    display_name = POSTURE_DISPLAY_NAMES.get(pred_class, pred_class)
    print(f"\nPosture:     {display_name}")
    print(f"Confidence:  {conf:.1%}")
    print(f"Risk Level:  {risk.upper()}")
    
    print("\nAll Class Probabilities:")
    for i, cls in enumerate(POSTURE_CLASSES):
        prob = probs[0, i].item()
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        display = POSTURE_DISPLAY_NAMES.get(cls, cls)
        print(f"  {display:20} {bar} {prob:.1%}")
    
    print("\nKey Measurements:")
    print(f"  Spine angle from vertical: {np.degrees(debug_info['spine_angle']):.1f}°")
    print(f"  Head forward position:     {debug_info['head_forward']:.3f}")
    
    # Simple visualization
    print("\nCreating visualization...")
    vis_image = extractor.visualize_side_view(image, debug_info['raw_keypoints'], debug_info)
    
    output_path = "prediction_result.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"✓ Visualization saved to: {output_path}")
    
    print("\n" + "="*70 + "\n")
    
    return {
        'class': pred_class,
        'confidence': conf,
        'risk': risk,
        'spine_angle': np.degrees(debug_info['spine_angle']),
        'head_forward': debug_info['head_forward']
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_demo.py <image_path>")
        print("\nExample:")
        print("  python simple_demo.py yolo_dataset/test/goodposture/img001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    simple_prediction(image_path)
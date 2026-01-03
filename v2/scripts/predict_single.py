"""
Single image inference script with visualization.
Predicts posture class, confidence, and provides visual feedback.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import (
    POSTURE_CLASSES,
    POSTURE_RISK_MAPPING,
    IDX_TO_CLASS,
    CONFIDENCE_THRESHOLD,
    get_device,
    get_model_save_path
)
from models.hybrid_model import create_model
from utils.pose_features import SideViewPoseFeatureExtractor
from utils.transforms import get_posture_transforms


class PosturePredictor:
    """
    Posture prediction class for single images.
    """
    
    def __init__(self, checkpoint_path: Optional[Path] = None, device: Optional[torch.device] = None):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint (uses best model if None)
            device: Device to run inference on (auto-detect if None)
        """
        self.device = device if device is not None else get_device()
        
        # Load model
        print("Loading model...")
        self.model = create_model(device=self.device)
        
        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = get_model_save_path(best=True)
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
            if 'best_val_acc' in checkpoint:
                print(f"Model validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Using randomly initialized model")
        
        self.model.eval()
        
        # Initialize pose extractor
        self.pose_extractor = SideViewPoseFeatureExtractor()
        
        # Get transforms
        self.transform = get_posture_transforms('test')
    
    def predict(self, image_path: str) -> Optional[Dict]:
        """
        Predict posture from an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results or None if pose detection fails
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Extract pose features
        result = self.pose_extractor.extract_features(image)
        if result is None:
            return None
        
        features, debug_info = result
        
        # Prepare image tensor
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Prepare pose tensor
        pose_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor, pose_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get results
        pred_idx = predicted.item()
        class_name = IDX_TO_CLASS[pred_idx]
        confidence_score = confidence.item()
        risk_level = POSTURE_RISK_MAPPING[class_name]
        
        # Compile results
        results = {
            'class': class_name,
            'confidence': confidence_score,
            'risk_level': risk_level,
            'probabilities': {
                POSTURE_CLASSES[i]: probabilities[0][i].item() 
                for i in range(len(POSTURE_CLASSES))
            },
            'spine_angle': debug_info.get('spine_angle', None),
            'head_forward_distance': debug_info.get('head_forward_distance', None),
            'detected_side': debug_info.get('side', None),
            'image': image,
            'landmarks': debug_info.get('landmarks', None),
            'pose_features': features
        }
        
        return results
    
    def visualize_prediction(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize prediction results.
        
        Args:
            results: Prediction results from predict()
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original image
        image_rgb = cv2.cvtColor(results['image'], cv2.COLOR_BGR2RGB)
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 2. Pose overlay
        if results['landmarks'] is not None:
            pose_image = self.pose_extractor.visualize_pose(
                results['image'],
                results['landmarks'],
                results['detected_side']
            )
            pose_image_rgb = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
            axes[1].imshow(pose_image_rgb)
        else:
            axes[1].imshow(image_rgb)
        
        axes[1].set_title(f'Detected Pose (Side: {results["detected_side"]})')
        axes[1].axis('off')
        
        # 3. Prediction results
        axes[2].axis('off')
        
        # Format text
        pred_text = f"PREDICTION RESULTS\n\n"
        pred_text += f"Class: {results['class'].upper()}\n"
        pred_text += f"Confidence: {results['confidence']*100:.1f}%\n"
        pred_text += f"Risk Level: {results['risk_level']}\n\n"
        pred_text += "Class Probabilities:\n"
        
        for class_name, prob in results['probabilities'].items():
            pred_text += f"  {class_name}: {prob*100:.1f}%\n"
        
        if results['spine_angle'] is not None:
            pred_text += f"\nSpine Angle: {results['spine_angle']:.1f}°\n"
        
        # Color based on risk
        risk_colors = {
            'Low Risk': 'green',
            'Medium Risk': 'orange',
            'High Risk': 'red'
        }
        text_color = risk_colors.get(results['risk_level'], 'black')
        
        axes[2].text(
            0.5, 0.5,
            pred_text,
            fontsize=12,
            ha='center',
            va='center',
            family='monospace',
            color=text_color,
            weight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def predict_and_visualize(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Dict]:
        """
        Predict and visualize in one call.
        
        Args:
            image_path: Path to image
            save_path: Optional path to save visualization
            show: Whether to display plot
            
        Returns:
            Prediction results or None
        """
        results = self.predict(image_path)
        
        if results is None:
            print("Error: Could not detect pose in image")
            return None
        
        self.visualize_prediction(results, save_path, show)
        
        return results


def main():
    """
    Example usage of PosturePredictor.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict posture from single image')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display visualization')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = PosturePredictor(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None
    )
    
    # Predict and visualize
    results = predictor.predict_and_visualize(
        args.image_path,
        save_path=args.output,
        show=not args.no_show
    )
    
    if results:
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Class: {results['class']}")
        print(f"Confidence: {results['confidence']*100:.1f}%")
        print(f"Risk Level: {results['risk_level']}")
        print(f"Spine Angle: {results['spine_angle']:.1f}°")
        print("="*60)


if __name__ == "__main__":
    main()
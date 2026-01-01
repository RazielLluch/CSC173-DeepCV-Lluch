"""
predict_single.py
Predict posture and ergonomic risk for a single image.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

from config import (
    POSTURE_CLASSES, CLASSIFIER_CONFIG, POSTURE_CLASSIFIER_PATH,
    POSTURE_DISPLAY_NAMES, POSTURE_RISK_MAPPING, POSTURE_COLORS,
    RISK_COLORS
)
from datasets import get_posture_transforms
from train_classifier import PostureClassifier


class PosturePredictor:
    """Class for predicting posture from single images."""
    
    def __init__(self, model_path: Path = POSTURE_CLASSIFIER_PATH, device: str = "cuda"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                "Please train a model first using: python train_classifier.py"
            )
        
        print(f"Loading model from {model_path}...")
        self.model = PostureClassifier(
            num_classes=len(POSTURE_CLASSES),
            backbone=CLASSIFIER_CONFIG["backbone"],
            pretrained=False
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get transform
        self.transform = get_posture_transforms("test")
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Best validation accuracy: {checkpoint.get('val_acc', 0):.2f}%\n")
    
    def predict(self, image_path: str):
        """
        Predict posture from image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with prediction results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Load and preprocess image
        pil_image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(pil_image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get results
        predicted_class = POSTURE_CLASSES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        class_probabilities = {
            POSTURE_CLASSES[i]: float(all_probs[i]) 
            for i in range(len(POSTURE_CLASSES))
        }
        
        # Get display name and risk
        display_name = POSTURE_DISPLAY_NAMES.get(predicted_class, predicted_class)
        risk_level = POSTURE_RISK_MAPPING.get(predicted_class, "medium")
        
        return {
            'image_path': str(image_path),
            'predicted_class': predicted_class,
            'display_name': display_name,
            'confidence': confidence_score,
            'risk_level': risk_level,
            'all_probabilities': class_probabilities
        }
    
    def predict_and_visualize(self, image_path: str, save_path: str = None, show: bool = True):
        """
        Predict and create visualization.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
            show: Whether to display the image
        
        Returns:
            Annotated image and prediction results
        """
        # Get prediction
        results = self.predict(image_path)
        
        # Load image with OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize if too large (for display purposes)
        max_height = 800
        h, w = image.shape[:2]
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, max_height))
        
        # Create info panel
        panel_height = 200
        panel = np.zeros((panel_height, image.shape[1], 3), dtype=np.uint8)
        
        # Get colors
        posture_color = POSTURE_COLORS.get(results['predicted_class'], (255, 255, 255))
        risk_color = RISK_COLORS.get(results['risk_level'], (255, 255, 255))
        
        # Draw prediction info
        y_pos = 40
        cv2.putText(panel, "POSTURE PREDICTION", (10, y_pos),
                   cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        
        y_pos += 40
        cv2.putText(panel, f"Class: {results['display_name']}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
        
        y_pos += 35
        cv2.putText(panel, f"Confidence: {results['confidence']:.1%}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 35
        cv2.putText(panel, f"Risk Level: {results['risk_level'].upper()}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        
        # Draw probability bars on the right side
        bar_x = image.shape[1] - 300
        bar_y_start = 30
        bar_height = 20
        bar_max_width = 250
        
        for i, class_name in enumerate(POSTURE_CLASSES):
            prob = results['all_probabilities'][class_name]
            display_name = POSTURE_DISPLAY_NAMES.get(class_name, class_name)
            
            y = bar_y_start + i * 50
            
            # Class name
            cv2.putText(panel, f"{display_name}:", (bar_x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Probability bar
            bar_width = int(bar_max_width * prob)
            bar_color = POSTURE_COLORS.get(class_name, (100, 100, 100))
            cv2.rectangle(panel, (bar_x, y + 5), (bar_x + bar_width, y + 5 + bar_height),
                         bar_color, -1)
            cv2.rectangle(panel, (bar_x, y + 5), (bar_x + bar_max_width, y + 5 + bar_height),
                         (100, 100, 100), 1)
            
            # Percentage text
            cv2.putText(panel, f"{prob:.1%}", (bar_x + bar_max_width + 10, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine image and panel
        result_image = np.vstack([image, panel])
        
        # Add border based on risk level
        border_size = 10
        result_image = cv2.copyMakeBorder(
            result_image, border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, value=risk_color
        )
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), result_image)
            print(f"✓ Visualization saved to {save_path}")
        
        # Display if requested
        if show:
            window_name = f"Posture Prediction - {results['display_name']}"
            cv2.imshow(window_name, result_image)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result_image, results


def print_prediction_results(results: dict):
    """Print prediction results in a nice format."""
    print("\n" + "="*70)
    print(" " * 20 + "PREDICTION RESULTS")
    print("="*70)
    print(f"\nImage: {results['image_path']}")
    print(f"\n{'Predicted Posture:':<25} {results['display_name']}")
    print(f"{'Confidence:':<25} {results['confidence']:.1%}")
    print(f"{'Ergonomic Risk Level:':<25} {results['risk_level'].upper()}")
    
    print("\n" + "-"*70)
    print("All Class Probabilities:")
    print("-"*70)
    
    for class_name in POSTURE_CLASSES:
        prob = results['all_probabilities'][class_name]
        display_name = POSTURE_DISPLAY_NAMES.get(class_name, class_name)
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{display_name:<25} {bar} {prob:.1%}")
    
    print("="*70 + "\n")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Predict posture from a single image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict and show visualization
  python predict_single.py path/to/image.jpg
  
  # Predict and save visualization
  python predict_single.py path/to/image.jpg --save results/prediction.jpg
  
  # Just predict, no visualization
  python predict_single.py path/to/image.jpg --no-show --no-save
  
  # Use CPU instead of GPU
  python predict_single.py path/to/image.jpg --device cpu
        """
    )
    
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization (optional)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display visualization window')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights (default: use config path)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print(" " * 15 + "GAMER ERGOVISION - SINGLE IMAGE PREDICTION")
    print("="*70 + "\n")
    
    # Check CUDA
    if args.device == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}\n")
        else:
            print("CUDA not available, falling back to CPU\n")
            args.device = 'cpu'
    
    # Initialize predictor
    model_path = Path(args.model) if args.model else POSTURE_CLASSIFIER_PATH
    predictor = PosturePredictor(model_path=model_path, device=args.device)
    
    # Determine save path
    save_path = args.save
    if save_path is None and not args.no_save:
        # Auto-generate save path
        input_path = Path(args.image_path)
        save_path = input_path.parent / f"{input_path.stem}_prediction.jpg"
    
    # Run prediction
    try:
        result_image, results = predictor.predict_and_visualize(
            args.image_path,
            save_path=save_path if not args.no_save else None,
            show=not args.no_show
        )
        
        # Print results
        print_prediction_results(results)
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
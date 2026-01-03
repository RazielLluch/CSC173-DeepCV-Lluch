"""
inference_pipeline.py
Complete inference pipeline for Gamer ErgoVision.
Runs detection, posture classification, distance estimation, and risk assessment.
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from config import (
    DETECTION_MODEL_PATH, POSTURE_CLASSIFIER_PATH, POSTURE_CLASSES,
    DETECTION_CLASSES, INFERENCE_CONFIG, DISTANCE_THRESHOLDS,
    RISK_MAPPING, DEFAULT_RISK_LEVEL, COLORS, RISK_COLORS,
    CLASSIFIER_CONFIG
)
from train_posture_classifier import PostureClassifier
from datasets import get_posture_transforms


class GamerErgoVision:
    """
    Complete inference pipeline for Gamer ErgoVision system.
    Detects person and monitor, classifies posture, estimates distance, and assesses ergonomic risk.
    """
    
    def __init__(
        self,
        detection_model_path: Path = DETECTION_MODEL_PATH,
        classifier_model_path: Path = POSTURE_CLASSIFIER_PATH,
        device: str = "cuda"
    ):
        """
        Initialize the ErgoVision pipeline.
        
        Args:
            detection_model_path: Path to trained YOLOv8 detection model
            classifier_model_path: Path to trained posture classifier
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load detection model
        print(f"Loading detection model from {detection_model_path}")
        if not detection_model_path.exists():
            raise FileNotFoundError(f"Detection model not found at {detection_model_path}")
        self.detector = YOLO(str(detection_model_path))
        
        # Load posture classifier
        print(f"Loading posture classifier from {classifier_model_path}")
        if not classifier_model_path.exists():
            raise FileNotFoundError(f"Classifier not found at {classifier_model_path}")
        
        self.classifier = PostureClassifier(
            num_classes=len(POSTURE_CLASSES),
            backbone=CLASSIFIER_CONFIG["backbone"],
            pretrained=False
        )
        checkpoint = torch.load(classifier_model_path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        # Get classifier transform
        self.classifier_transform = get_posture_transforms("val")
        
        print("Pipeline initialized successfully!\n")
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Run YOLOv8 detection to find person and monitor.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detections with class, bbox, and confidence
        """
        results = self.detector.predict(
            image,
            conf=INFERENCE_CONFIG["detection_conf_threshold"],
            iou=INFERENCE_CONFIG["nms_iou_threshold"],
            max_det=INFERENCE_CONFIG["max_det"],
            verbose=False
        )
        
        detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                
                if class_id < len(DETECTION_CLASSES):
                    detections.append({
                        'class': DETECTION_CLASSES[class_id],
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': bbox.tolist()
                    })
        
        return detections
    
    def classify_posture(self, person_crop: np.ndarray) -> Tuple[str, float]:
        """
        Classify posture from person crop.
        
        Args:
            person_crop: Cropped image of person (BGR format)
        
        Returns:
            Tuple of (posture_class, confidence)
        """
        # Convert BGR to RGB and create PIL image
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
        
        # Apply transforms
        input_tensor = self.classifier_transform(pil_image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        posture_class = POSTURE_CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        return posture_class, confidence_score
    
    def estimate_distance(
        self,
        person_bbox: List[float],
        monitor_bbox: List[float],
        image_width: int
    ) -> Tuple[str, float]:
        """
        Estimate relative head-to-screen distance from bounding boxes.
        
        This is a simple geometric proxy based on:
        1. Horizontal distance between person and monitor centers
        2. Normalized by image width
        3. Adjusted by relative sizes (larger person = closer to camera)
        
        TODO: This is a simplified heuristic. You can refine this by:
        - Using depth estimation models
        - Calibrating with known distances
        - Adding more geometric features
        
        Args:
            person_bbox: Person bounding box [x1, y1, x2, y2]
            monitor_bbox: Monitor bounding box [x1, y1, x2, y2]
            image_width: Width of input image
        
        Returns:
            Tuple of (distance_bin, distance_score)
            distance_bin: "too_close", "acceptable", or "far"
            distance_score: Normalized distance metric [0, 1]
        """
        # Calculate centers
        person_center_x = (person_bbox[0] + person_bbox[2]) / 2
        monitor_center_x = (monitor_bbox[0] + monitor_bbox[2]) / 2
        
        # Calculate box sizes (area as proxy for perceived size)
        person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])
        monitor_area = (monitor_bbox[2] - monitor_bbox[0]) * (monitor_bbox[3] - monitor_bbox[1])
        
        # Horizontal distance (absolute)
        horizontal_distance = abs(person_center_x - monitor_center_x)
        
        # Normalize by image width
        normalized_distance = horizontal_distance / image_width
        
        # Adjust by size ratio (larger person relative to monitor = closer)
        # This is a heuristic: if person appears large, they're likely close to camera
        size_ratio = person_area / (monitor_area + 1e-6)  # Avoid division by zero
        
        # Combined distance score (higher = farther)
        # Weight normalized distance more heavily
        distance_score = 0.7 * normalized_distance + 0.3 * (1.0 / (1.0 + size_ratio))
        
        # Clip to [0, 1]
        distance_score = np.clip(distance_score, 0.0, 1.0)
        
        # Bin into categories
        if distance_score < DISTANCE_THRESHOLDS["too_close"]:
            distance_bin = "too_close"
        elif distance_score < DISTANCE_THRESHOLDS["acceptable"]:
            distance_bin = "acceptable"
        else:
            distance_bin = "far"
        
        return distance_bin, distance_score
    
    def assess_ergonomic_risk(
        self,
        posture_class: str,
        distance_bin: str
    ) -> str:
        """
        Map posture and distance to ergonomic risk level.
        
        Args:
            posture_class: Detected posture class
            distance_bin: Distance category
        
        Returns:
            Risk level: "low", "medium", or "high"
        """
        risk_key = (posture_class, distance_bin)
        risk_level = RISK_MAPPING.get(risk_key, DEFAULT_RISK_LEVEL)
        
        return risk_level
    
    def process_image(self, image_path: str) -> Dict:
        """
        Complete pipeline: detect, classify, estimate, and assess.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Dictionary with all results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        h, w = image.shape[:2]
        
        # Step 1: Detect objects
        detections = self.detect_objects(image)
        
        # Find person and monitor
        person_det = None
        monitor_det = None
        
        for det in detections:
            if det['class'] == 'person' and person_det is None:
                person_det = det
            elif det['class'] in ['desktop_monitor', 'laptop_screen'] and monitor_det is None:
                monitor_det = det
        
        # Check if we have both detections
        if person_det is None:
            return {
                'success': False,
                'error': 'No person detected in image',
                'detections': detections
            }
        
        if monitor_det is None:
            return {
                'success': False,
                'error': 'No monitor/screen detected in image',
                'detections': detections
            }
        
        # Step 2: Classify posture
        person_bbox = person_det['bbox']
        x1, y1, x2, y2 = map(int, person_bbox)
        person_crop = image[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return {
                'success': False,
                'error': 'Invalid person crop',
                'detections': detections
            }
        
        posture_class, posture_conf = self.classify_posture(person_crop)
        
        # Step 3: Estimate distance
        distance_bin, distance_score = self.estimate_distance(
            person_bbox, monitor_det['bbox'], w
        )
        
        # Step 4: Assess risk
        risk_level = self.assess_ergonomic_risk(posture_class, distance_bin)
        
        # Compile results
        result = {
            'success': True,
            'detections': detections,
            'person': {
                'bbox': person_bbox,
                'confidence': person_det['confidence']
            },
            'monitor': {
                'class': monitor_det['class'],
                'bbox': monitor_det['bbox'],
                'confidence': monitor_det['confidence']
            },
            'posture': {
                'class': posture_class,
                'confidence': posture_conf
            },
            'distance': {
                'bin': distance_bin,
                'score': distance_score
            },
            'ergonomic_risk': {
                'level': risk_level,
                'factors': {
                    'posture': posture_class,
                    'distance': distance_bin
                }
            }
        }
        
        return result
    
    def visualize_results(
        self,
        image_path: str,
        results: Dict,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detection and assessment results on image.
        
        Args:
            image_path: Path to input image
            results: Results dictionary from process_image()
            save_path: Optional path to save visualization
        
        Returns:
            Annotated image
        """
        image = cv2.imread(image_path)
        
        if not results['success']:
            # Draw error message
            cv2.putText(image, f"Error: {results['error']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image
        
        # Draw person bbox
        person_bbox = results['person']['bbox']
        x1, y1, x2, y2 = map(int, person_bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS['person'], 2)
        cv2.putText(image, f"Person: {results['person']['confidence']:.2f}",
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['person'], 2)
        
        # Draw monitor bbox
        monitor_bbox = results['monitor']['bbox']
        x1, y1, x2, y2 = map(int, monitor_bbox)
        monitor_class = results['monitor']['class']
        color = COLORS.get(monitor_class, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{monitor_class}: {results['monitor']['confidence']:.2f}",
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw info panel
        panel_height = 150
        panel = np.zeros((panel_height, image.shape[1], 3), dtype=np.uint8)
        
        # Risk level (large text)
        risk_level = results['ergonomic_risk']['level'].upper()
        risk_color = RISK_COLORS[results['ergonomic_risk']['level']]
        cv2.putText(panel, f"Risk: {risk_level}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, risk_color, 3)
        
        # Posture info
        posture = results['posture']['class']
        posture_conf = results['posture']['confidence']
        cv2.putText(panel, f"Posture: {posture} ({posture_conf:.2f})", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Distance info
        distance_bin = results['distance']['bin']
        distance_score = results['distance']['score']
        cv2.putText(panel, f"Distance: {distance_bin} ({distance_score:.2f})", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine image and panel
        result_image = np.vstack([image, panel])
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"Visualization saved to {save_path}")
        
        return result_image


def demo_inference(image_path: str, save_visualization: bool = True):
    """
    Demo function to run inference on a single image.
    
    Args:
        image_path: Path to test image
        save_visualization: Whether to save visualization
    """
    print(f"\n{'='*60}")
    print("Gamer ErgoVision - Demo Inference")
    print(f"{'='*60}\n")
    
    # Initialize pipeline
    pipeline = GamerErgoVision()
    
    # Process image
    print(f"Processing image: {image_path}")
    results = pipeline.process_image(image_path)
    
    # Print results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    # Visualize
    if save_visualization:
        save_path = Path(image_path).parent / f"{Path(image_path).stem}_result.jpg"
        pipeline.visualize_results(image_path, results, str(save_path))
    
    return results


# Main script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Gamer ErgoVision inference")
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--no-vis', action='store_true',
                       help='Skip visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Run demo
    results = demo_inference(args.image_path, save_visualization=not args.no_vis)
    
    print("\nDone!")
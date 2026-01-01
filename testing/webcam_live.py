"""
webcam_live.py
Live webcam feed with real-time posture detection and classification.
Uses YOLOv8 for person detection + custom classifier for posture.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from collections import deque

from ..config import (
    POSTURE_CLASSES, POSTURE_DISPLAY_NAMES, POSTURE_RISK_MAPPING,
    POSTURE_COLORS, RISK_COLORS, POSTURE_CLASSIFIER_PATH
)
from predict_single import PosturePredictor


class LivePostureDetector:
    """Real-time posture detection from webcam feed."""
    
    def __init__(
        self,
        camera_id: int = 0,
        detection_method: str = "yolo",
        predictor: PosturePredictor = None,
        confidence_threshold: float = 0.5,
        smooth_predictions: int = 5
    ):
        """
        Initialize live detector.
        
        Args:
            camera_id: Webcam ID (0 for default)
            detection_method: "yolo" for YOLOv8, "opencv" for OpenCV cascade
            predictor: PosturePredictor instance
            confidence_threshold: Minimum confidence for person detection
            smooth_predictions: Number of frames to smooth predictions (reduces jitter)
        """
        self.camera_id = camera_id
        self.detection_method = detection_method
        self.confidence_threshold = confidence_threshold
        
        # Initialize predictor
        if predictor is None:
            print("Initializing posture classifier...")
            self.predictor = PosturePredictor()
        else:
            self.predictor = predictor
        
        # Initialize person detector
        print(f"Initializing person detector ({detection_method})...")
        if detection_method == "yolo":
            from ultralytics import YOLO
            # Use a lightweight YOLOv8 nano model for person detection
            self.person_detector = YOLO('yolov8n.pt')  # Will auto-download
            print("✓ YOLOv8 person detector loaded")
        elif detection_method == "opencv":
            # OpenCV Haar Cascade (faster but less accurate)
            cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            self.person_detector = cv2.CascadeClassifier(cascade_path)
            if self.person_detector.empty():
                raise ValueError("Could not load OpenCV cascade classifier")
            print("✓ OpenCV cascade detector loaded")
        
        # Prediction smoothing (rolling average)
        self.smooth_predictions = smooth_predictions
        self.prediction_buffer = deque(maxlen=smooth_predictions)
        
        # FPS calculation
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
    
    def detect_person_yolo(self, frame):
        """
        Detect person using YOLOv8.
        
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        results = self.person_detector.predict(
            frame,
            classes=[0],  # Person class in COCO dataset
            conf=self.confidence_threshold,
            verbose=False
        )
        
        bboxes = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                bboxes.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence
                })
        
        return bboxes
    
    def detect_person_opencv(self, frame):
        """
        Detect person using OpenCV cascade.
        
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = self.person_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(100, 100)
        )
        
        bboxes = []
        for (x, y, w, h) in bodies:
            bboxes.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 1.0  # OpenCV doesn't provide confidence
            })
        
        return bboxes
    
    def smooth_prediction(self, current_prediction):
        """
        Smooth predictions over multiple frames to reduce jitter.
        
        Args:
            current_prediction: Current frame's prediction dict
        
        Returns:
            Smoothed prediction dict
        """
        self.prediction_buffer.append(current_prediction)
        
        if len(self.prediction_buffer) < 2:
            return current_prediction
        
        # Average probabilities across buffer
        avg_probs = {}
        for class_name in POSTURE_CLASSES:
            probs = [p['all_probabilities'][class_name] for p in self.prediction_buffer]
            avg_probs[class_name] = np.mean(probs)
        
        # Get class with highest average probability
        predicted_class = max(avg_probs, key=avg_probs.get)
        
        # Create smoothed result
        smoothed_result = {
            'predicted_class': predicted_class,
            'display_name': POSTURE_DISPLAY_NAMES.get(predicted_class, predicted_class),
            'confidence': avg_probs[predicted_class],
            'risk_level': POSTURE_RISK_MAPPING.get(predicted_class, 'medium'),
            'all_probabilities': avg_probs
        }
        
        return smoothed_result
    
    def calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer)
    
    def draw_info_panel(self, frame, prediction, fps, detection_bbox=None):
        """
        Draw information panel on frame.
        
        Args:
            frame: Video frame
            prediction: Prediction results dict
            fps: Current FPS
            detection_bbox: Person detection bbox (for confidence)
        """
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for info panel
        overlay = frame.copy()
        panel_height = 180
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Get colors
        risk_color = RISK_COLORS.get(prediction['risk_level'], (255, 255, 255))
        posture_color = POSTURE_COLORS.get(prediction['predicted_class'], (255, 255, 255))
        
        # Draw title
        cv2.putText(frame, "LIVE POSTURE DETECTION", (10, 30),
                   cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw posture info
        y_pos = 70
        cv2.putText(frame, f"Posture: {prediction['display_name']}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
        
        y_pos += 35
        cv2.putText(frame, f"Confidence: {prediction['confidence']:.1%}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 35
        cv2.putText(frame, f"Risk: {prediction['risk_level'].upper()}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        
        # Draw probability bars (compact)
        bar_x = w - 350
        bar_y = 70
        bar_width = 300
        bar_height = 15
        
        for i, class_name in enumerate(POSTURE_CLASSES):
            prob = prediction['all_probabilities'][class_name]
            display_name = POSTURE_DISPLAY_NAMES.get(class_name, class_name)
            
            y = bar_y + i * 30
            
            # Class name
            cv2.putText(frame, f"{display_name}:", (bar_x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Probability bar
            filled_width = int(bar_width * prob)
            color = POSTURE_COLORS.get(class_name, (100, 100, 100))
            cv2.rectangle(frame, (bar_x, y + 5), (bar_x + filled_width, y + 5 + bar_height),
                         color, -1)
            cv2.rectangle(frame, (bar_x, y + 5), (bar_x + bar_width, y + 5 + bar_height),
                         (100, 100, 100), 1)
    
    def draw_detection_box(self, frame, bbox, confidence, prediction):
        """
        Draw bounding box around detected person.
        
        Args:
            frame: Video frame
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence
            prediction: Posture prediction results
        """
        x1, y1, x2, y2 = bbox
        
        # Color based on risk level
        risk_color = RISK_COLORS.get(prediction['risk_level'], (255, 255, 255))
        
        # Draw thick bounding box
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), risk_color, thickness)
        
        # Draw label background
        label = f"{prediction['display_name']} ({prediction['confidence']:.0%})"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1),
                     risk_color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw risk indicator in corner of bbox
        risk_text = prediction['risk_level'].upper()
        cv2.putText(frame, risk_text, (x2 - 80, y2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)
    
    def run(self, display_width: int = 1280, save_video: str = None):
        """
        Run live detection loop.
        
        Args:
            display_width: Width to resize display (maintains aspect ratio)
            save_video: Optional path to save video output
        """
        # Open camera
        print(f"\nOpening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        # Get camera properties
        cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ Camera opened: {cam_width}x{cam_height} @ {cam_fps} FPS")
        
        # Video writer setup
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_video, fourcc, 20.0, (display_width, 
                int(cam_height * display_width / cam_width)))
            print(f"✓ Recording to {save_video}")
        
        print("\n" + "="*70)
        print("LIVE POSTURE DETECTION ACTIVE")
        print("="*70)
        print("\nControls:")
        print("  'q' or ESC - Quit")
        print("  's' - Save screenshot")
        print("  'r' - Reset prediction smoothing")
        print("  SPACE - Pause/Resume")
        print("\nDetecting posture...\n")
        
        paused = False
        last_prediction = None
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        break
                    
                    self.frame_count += 1
                    
                    # Detect person
                    if self.detection_method == "yolo":
                        detections = self.detect_person_yolo(frame)
                    else:
                        detections = self.detect_person_opencv(frame)
                    
                    # Process first detection
                    if detections:
                        self.detection_count += 1
                        detection = detections[0]  # Use first detection
                        bbox = detection['bbox']
                        
                        # Crop person region
                        x1, y1, x2, y2 = bbox
                        person_crop = frame[y1:y2, x1:x2]
                        
                        if person_crop.size > 0:
                            # Classify posture
                            prediction = self.predictor.predict_from_array(person_crop)
                            
                            # Smooth prediction
                            prediction = self.smooth_prediction(prediction)
                            last_prediction = prediction
                            
                            # Draw detection box
                            self.draw_detection_box(frame, bbox, detection['confidence'], 
                                                    prediction)
                    else:
                        # No detection - show last prediction if available
                        if last_prediction:
                            prediction = last_prediction
                        else:
                            # No person detected yet
                            cv2.putText(frame, "No person detected", 
                                      (frame.shape[1]//2 - 150, frame.shape[0]//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                            prediction = None
                    
                    # Calculate FPS
                    fps = self.calculate_fps()
                    
                    # Draw info panel
                    if prediction:
                        self.draw_info_panel(frame, prediction, fps)
                
                # Resize for display
                display_height = int(frame.shape[0] * display_width / frame.shape[1])
                display_frame = cv2.resize(frame, (display_width, display_height))
                
                # Show frame
                cv2.imshow('Gamer ErgoVision - Live Detection', display_frame)
                
                # Save video frame
                if video_writer and not paused:
                    video_writer.write(display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"screenshots/screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, display_frame)
                    print(f"✓ Screenshot saved: {screenshot_path}")
                elif key == ord('r'):  # Reset smoothing
                    self.prediction_buffer.clear()
                    print("✓ Prediction smoothing reset")
                elif key == ord(' '):  # Pause/Resume
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"✓ {status}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            print("\n" + "="*70)
            print("SESSION STATISTICS")
            print("="*70)
            print(f"Total frames: {self.frame_count}")
            print(f"Frames with detection: {self.detection_count}")
            if self.frame_count > 0:
                detection_rate = (self.detection_count / self.frame_count) * 100
                print(f"Detection rate: {detection_rate:.1f}%")
            print("="*70 + "\n")


# Extend PosturePredictor with array input
def predict_from_array(self, image_array):
    """Predict posture from numpy array (for webcam frames)."""
    from PIL import Image
    
    # Convert BGR to RGB
    rgb_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_array)
    
    # Preprocess
    input_tensor = self.transform(pil_image).unsqueeze(0)
    input_tensor = input_tensor.to(self.device)
    
    # Inference
    with torch.no_grad():
        outputs = self.model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    # Results
    predicted_class = POSTURE_CLASSES[predicted_idx.item()]
    confidence_score = confidence.item()
    
    all_probs = probabilities[0].cpu().numpy()
    class_probabilities = {
        POSTURE_CLASSES[i]: float(all_probs[i]) 
        for i in range(len(POSTURE_CLASSES))
    }
    
    display_name = POSTURE_DISPLAY_NAMES.get(predicted_class, predicted_class)
    risk_level = POSTURE_RISK_MAPPING.get(predicted_class, "medium")
    
    return {
        'predicted_class': predicted_class,
        'display_name': display_name,
        'confidence': confidence_score,
        'risk_level': risk_level,
        'all_probabilities': class_probabilities
    }

# Add method to PosturePredictor class
PosturePredictor.predict_from_array = predict_from_array


def main():
    parser = argparse.ArgumentParser(
        description="Live webcam posture detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default camera
  python webcam_live.py
  
  # Use specific camera
  python webcam_live.py --camera 1
  
  # Use OpenCV detector (faster)
  python webcam_live.py --detector opencv
  
  # Save video output
  python webcam_live.py --save output.mp4
  
  # Lower display resolution for performance
  python webcam_live.py --width 960

Controls:
  'q' or ESC - Quit
  's' - Save screenshot
  'r' - Reset prediction smoothing
  SPACE - Pause/Resume
        """
    )
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--detector', type=str, default='yolo',
                       choices=['yolo', 'opencv'],
                       help='Person detection method (default: yolo)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Display width in pixels (default: 1280)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save video to file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--smooth', type=int, default=5,
                       help='Prediction smoothing frames (default: 5)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference (default: cuda)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print(" " * 15 + "GAMER ERGOVISION - LIVE WEBCAM DETECTION")
    print("="*70 + "\n")
    
    # Check CUDA
    if args.device == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")
            args.device = 'cpu'
    print()
    
    # Initialize predictor
    predictor = PosturePredictor(device=args.device)
    
    # Initialize and run detector
    detector = LivePostureDetector(
        camera_id=args.camera,
        detection_method=args.detector,
        predictor=predictor,
        confidence_threshold=args.confidence,
        smooth_predictions=args.smooth
    )
    
    detector.run(display_width=args.width, save_video=args.save)


if __name__ == "__main__":
    main()
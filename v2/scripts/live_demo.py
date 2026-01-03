"""
Real-time Posture Classification Demo with Webcam
Shows live camera feed with:
- MediaPipe pose skeleton overlay
- Real-time posture classification
- Confidence scores for all classes
- Visual feedback with color-coded risk levels
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ..config import (
    POSTURE_CLASSES,
    POSTURE_RISK_MAPPING,
    get_device,
    get_model_save_path
)
from ..models.hybrid_model import create_model
from ..utils.pose_features import SideViewPoseFeatureExtractor
from ..utils.transforms import get_posture_transforms

# MediaPipe imports
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style
except ImportError:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils


class RealtimePostureClassifier:
    """
    Real-time posture classification from webcam feed.
    """
    
    def __init__(
        self,
        checkpoint_path: Path = None,
        camera_id: int = 0,
        show_fps: bool = True,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize real-time classifier.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            camera_id: Camera device ID (0 for default webcam)
            show_fps: Whether to display FPS counter
            confidence_threshold: Minimum confidence to show prediction
        """
        self.camera_id = camera_id
        self.show_fps = show_fps
        self.confidence_threshold = confidence_threshold
        
        # Setup device
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = create_model(device=self.device)
        
        if checkpoint_path is None:
            checkpoint_path = get_model_save_path(best=True)
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from {checkpoint_path}")
            if 'best_val_acc' in checkpoint:
                print(f"   Model accuracy: {checkpoint['best_val_acc']:.2f}%")
        else:
            print(f"⚠️  No checkpoint found, using random weights (demo only)")
        
        self.model.eval()
        
        # Initialize pose extractor
        print("Initializing pose detector...")
        self.pose_extractor = SideViewPoseFeatureExtractor()
        
        # Get image transforms
        self.transform = get_posture_transforms('test')
        
        # Initialize MediaPipe pose for drawing
        self.mp_pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera
        print(f"Opening camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # FPS tracking
        self.fps = 0
        self.fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        # Color scheme
        self.colors = {
            'goodposture': (0, 255, 0),        # Green
            'forwardbadposture': (0, 0, 255),  # Red
            'backwardbadposture': (0, 165, 255) # Orange
        }
        
        print("✅ Initialization complete!")
        print("\nControls:")
        print("  'q' or 'ESC' - Quit")
        print("  'f' - Toggle FPS display")
        print("  's' - Save screenshot")
        print()
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame and get predictions.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Tuple of (predictions, pose_landmarks, debug_info)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract pose features
        result = self.pose_extractor.extract_features(rgb_frame)
        
        if result is None:
            return None, None, None
        
        features, debug_info = result
        
        # Prepare image tensor
        pil_image = Image.fromarray(rgb_frame)
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Prepare pose tensor
        pose_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(image_tensor, pose_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Get predictions
        probs = probabilities[0].cpu().numpy()
        pred_idx = np.argmax(probs)
        
        predictions = {
            'class': POSTURE_CLASSES[pred_idx],
            'confidence': probs[pred_idx],
            'probabilities': {POSTURE_CLASSES[i]: probs[i] for i in range(len(POSTURE_CLASSES))},
            'risk_level': POSTURE_RISK_MAPPING[POSTURE_CLASSES[pred_idx]]
        }
        
        return predictions, debug_info['landmarks'], debug_info
    
    def draw_pose_skeleton(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """
        Draw MediaPipe pose skeleton on frame.
        
        Args:
            frame: BGR image
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Frame with skeleton overlay
        """
        if landmarks is None:
            return frame
        
        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255), thickness=2
            )
        )
        
        return frame
    
    def draw_predictions(self, frame: np.ndarray, predictions: dict, debug_info: dict) -> np.ndarray:
        """
        Draw prediction information on frame.
        
        Args:
            frame: BGR image
            predictions: Prediction dictionary
            debug_info: Debug information with metrics
            
        Returns:
            Frame with predictions overlay
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay panel
        overlay = frame.copy()
        panel_width = 350
        panel_height = 280
        
        # Draw panel background
        cv2.rectangle(
            overlay,
            (w - panel_width - 20, 20),
            (w - 20, panel_height),
            (40, 40, 40),
            -1
        )
        
        # Blend with original
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Get predicted class and color
        pred_class = predictions['class']
        confidence = predictions['confidence']
        risk_level = predictions['risk_level']
        color = self.colors.get(pred_class, (255, 255, 255))
        
        # Draw title
        cv2.putText(
            frame,
            "POSTURE ANALYSIS",
            (w - panel_width, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw separator
        cv2.line(
            frame,
            (w - panel_width, 60),
            (w - 30, 60),
            (100, 100, 100),
            1
        )
        
        # Draw predicted class
        y_offset = 85
        cv2.putText(
            frame,
            f"Class: {pred_class.upper()}",
            (w - panel_width, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        # Draw confidence
        y_offset += 25
        cv2.putText(
            frame,
            f"Confidence: {confidence*100:.1f}%",
            (w - panel_width, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Draw risk level with color
        y_offset += 25
        risk_color = {
            'Low Risk': (0, 255, 0),
            'Medium Risk': (0, 165, 255),
            'High Risk': (0, 0, 255)
        }.get(risk_level, (255, 255, 255))
        
        cv2.putText(
            frame,
            f"Risk: {risk_level}",
            (w - panel_width, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            risk_color,
            2
        )
        
        # Draw separator
        y_offset += 15
        cv2.line(
            frame,
            (w - panel_width, y_offset),
            (w - 30, y_offset),
            (100, 100, 100),
            1
        )
        
        # Draw all class probabilities
        y_offset += 25
        cv2.putText(
            frame,
            "Class Probabilities:",
            (w - panel_width, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        y_offset += 20
        for class_name in POSTURE_CLASSES:
            prob = predictions['probabilities'][class_name]
            class_color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw class name
            cv2.putText(
                frame,
                f"{class_name}:",
                (w - panel_width, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )
            
            # Draw progress bar
            bar_x = w - panel_width + 180
            bar_y = y_offset - 10
            bar_width = 120
            bar_height = 12
            
            # Background bar
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (60, 60, 60),
                -1
            )
            
            # Filled bar
            fill_width = int(bar_width * prob)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_width, bar_y + bar_height),
                class_color,
                -1
            )
            
            # Percentage text
            cv2.putText(
                frame,
                f"{prob*100:.1f}%",
                (bar_x + bar_width + 5, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
            
            y_offset += 20
        
        # Draw spine angle if available
        if 'spine_angle' in debug_info:
            y_offset += 10
            cv2.putText(
                frame,
                f"Spine Angle: {debug_info['spine_angle']:.1f}°",
                (w - panel_width, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )
        
        return frame
    
    def draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter on frame."""
        if not self.show_fps:
            return frame
        
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return frame
    
    def update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_update >= self.fps_update_interval:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def run(self):
        """
        Main loop for real-time classification.
        """
        print("🎥 Starting live demo...")
        print("Position yourself in side view for best results")
        print()
        
        screenshot_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                predictions, landmarks, debug_info = self.process_frame(frame)
                
                # Draw visualizations
                if predictions is not None and landmarks is not None:
                    # Draw pose skeleton
                    frame = self.draw_pose_skeleton(frame, landmarks)
                    
                    # Draw predictions
                    frame = self.draw_predictions(frame, predictions, debug_info)
                else:
                    # No pose detected - show message
                    h, w = frame.shape[:2]
                    cv2.putText(
                        frame,
                        "NO POSE DETECTED",
                        (w//2 - 150, h//2),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        "Position yourself in side view",
                        (w//2 - 180, h//2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        1
                    )
                
                # Draw FPS
                frame = self.draw_fps(frame)
                
                # Update FPS counter
                self.update_fps()
                
                # Display frame
                cv2.imshow('Gamer ErgoVision - Live Posture Classification', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nExiting...")
                    break
                elif key == ord('f'):  # Toggle FPS
                    self.show_fps = not self.show_fps
                    print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
                elif key == ord('s'):  # Save screenshot
                    screenshot_path = f"screenshot_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"💾 Saved screenshot: {screenshot_path}")
                    screenshot_count += 1
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            print("\nCleaning up...")
            self.cap.release()
            cv2.destroyAllWindows()
            if hasattr(self.pose_extractor, 'pose'):
                self.pose_extractor.pose.close()
            if hasattr(self, 'mp_pose'):
                self.mp_pose.close()
            
            print("✅ Demo ended")

def main():
    """Run the live demo."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time posture classification from webcam'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (default: checkpoints/best_model.pth)'
    )
    parser.add_argument(
        '--no-fps',
        action='store_true',
        help='Hide FPS counter'
    )
    
    args = parser.parse_args()
    
    # Create and run classifier
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    
    classifier = RealtimePostureClassifier(
        checkpoint_path=checkpoint_path,
        camera_id=args.camera,
        show_fps=not args.no_fps
    )
    
    classifier.run()


if __name__ == "__main__":
    main()
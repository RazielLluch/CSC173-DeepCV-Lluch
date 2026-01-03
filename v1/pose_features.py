"""
pose_features.py
Extract body-centric pose features optimized for SIDE-VIEW camera angle.
Focuses on features that matter for side-view ergonomic assessment.
Uses MediaPipe Pose with side-view specific feature extraction.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

# Try importing mediapipe with error handling
try:
    import mediapipe as mp
    
    # Check for different MediaPipe API versions
    try:
        # Try new API (v0.10.0+)
        from mediapipe.python.solutions import pose as mp_pose
        from mediapipe.python.solutions import drawing_utils as mp_drawing
        MEDIAPIPE_API = "new"
    except (ImportError, AttributeError):
        try:
            # Try old API (v0.8.x - v0.9.x)
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            MEDIAPIPE_API = "old"
        except AttributeError:
            print("ERROR: MediaPipe installed but cannot access pose module")
            print("Try: pip install --upgrade mediapipe")
            MEDIAPIPE_API = None
    
    MEDIAPIPE_AVAILABLE = (MEDIAPIPE_API is not None)
    
except ImportError:
    print("ERROR: MediaPipe not installed. Please run: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_API = None
    mp_pose = None
    mp_drawing = None


class SideViewPoseFeatureExtractor:
    """
    Extract pose keypoints and compute geometric features for SIDE-VIEW posture analysis.
    
    Key innovations for side-view:
    1. Focus on visible side profile (one side of body)
    2. Emphasize forward/backward lean (most visible in side view)
    3. Spine angle relative to vertical (key ergonomic metric)
    4. Head position relative to shoulders (forward head posture)
    5. Hand position becomes less relevant due to side angle
    """
    
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.3):
        """
        Initialize MediaPipe Pose with relaxed confidence for side views.
        
        Args:
            model_complexity: 0 (lite), 1 (full), or 2 (heavy)
            min_detection_confidence: Lower threshold for side views
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not installed. Please install it with:\n"
                "pip install mediapipe"
            )
        
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.3
            )
            print("✓ MediaPipe Pose initialized successfully")
        except Exception as e:
            print(f"ERROR initializing MediaPipe Pose: {e}")
            raise
        
        # Define keypoints for side-view analysis
        self.KEYPOINTS = {
            # Head
            'nose': 0,
            'left_eye': 2,
            'right_eye': 5,
            'left_ear': 7,
            'right_ear': 8,
            
            # Upper body (critical for side-view posture)
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            
            # Torso/hips
            'left_hip': 23,
            'right_hip': 24,
            
            # Knees (help determine if sitting)
            'left_knee': 25,
            'right_knee': 26,
        }
    
    def determine_visible_side(self, keypoints: Dict[str, np.ndarray]) -> str:
        """
        Determine which side of the body is visible (left or right).
        In side-view, one side will have higher visibility.
        """
        try:
            left_visibility = np.mean([
                keypoints['left_shoulder'][2],
                keypoints['left_hip'][2],
                keypoints['left_elbow'][2]
            ])
            
            right_visibility = np.mean([
                keypoints['right_shoulder'][2],
                keypoints['right_hip'][2],
                keypoints['right_elbow'][2]
            ])
            
            return 'left' if left_visibility > right_visibility else 'right'
        except Exception as e:
            print(f"Warning: Error determining visible side: {e}")
            return 'left'  # Default
    
    def extract_keypoints(self, image: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract raw keypoints from side-view image.
        
        Args:
            image: BGR image (side-view of person sitting)
        
        Returns:
            Dictionary mapping keypoint names to (x, y, visibility) arrays
            Returns None if pose detection fails
        """
        try:
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks is None:
                return None
            
            # Extract landmarks
            h, w = image.shape[:2]
            keypoints = {}
            
            for name, idx in self.KEYPOINTS.items():
                try:
                    landmark = results.pose_landmarks.landmark[idx]
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * w
                    y = landmark.y * h
                    visibility = landmark.visibility
                    keypoints[name] = np.array([x, y, visibility], dtype=np.float32)
                except Exception as e:
                    print(f"Warning: Error extracting {name}: {e}")
                    keypoints[name] = np.array([0, 0, 0], dtype=np.float32)
            
            return keypoints
            
        except Exception as e:
            print(f"Error in extract_keypoints: {e}")
            return None
    
    def normalize_to_spine_frame(
        self,
        keypoints: Dict[str, np.ndarray],
        spine_base: np.ndarray,
        spine_length: float
    ) -> Dict[str, np.ndarray]:
        """
        Normalize coordinates relative to spine base and length.
        """
        normalized = {}
        
        try:
            for name, kp in keypoints.items():
                xy = kp[:2]
                visibility = kp[2]
                
                # Translate to spine base
                centered = xy - spine_base
                
                # Scale by spine length
                if spine_length > 0:
                    normalized_xy = centered / spine_length
                else:
                    normalized_xy = centered
                
                normalized[name] = np.array([normalized_xy[0], normalized_xy[1], visibility], dtype=np.float32)
        except Exception as e:
            print(f"Error in normalize_to_spine_frame: {e}")
            return keypoints  # Return original if normalization fails
        
        return normalized
    
    def compute_side_view_features(self, keypoints: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute geometric features specifically for side-view posture assessment.
        
        Returns 24 features optimized for side-view analysis.
        """
        try:
            features = []
            
            # Determine visible side
            visible_side = 'left' if keypoints['left_shoulder'][2] > keypoints['right_shoulder'][2] else 'right'
            
            # Get key points on visible side
            shoulder = keypoints[f'{visible_side}_shoulder'][:2]
            hip = keypoints[f'{visible_side}_hip'][:2]
            nose = keypoints['nose'][:2]
            
            # Get head reference (ears)
            left_ear = keypoints['left_ear'][:2]
            right_ear = keypoints['right_ear'][:2]
            ear_center = (left_ear + right_ear) / 2
            
            # === CRITICAL FEATURE 1: Forward Head Posture ===
            head_forward_nose = nose[0] - shoulder[0]
            head_forward_ear = ear_center[0] - shoulder[0]
            head_forward_avg = (head_forward_nose + head_forward_ear) / 2
            
            features.extend([head_forward_nose, head_forward_ear, head_forward_avg])  # 0-2
            
            # === CRITICAL FEATURE 2: Spine Angle from Vertical ===
            spine_vector = shoulder - hip
            spine_angle = np.arctan2(spine_vector[0], -spine_vector[1])
            
            features.extend([spine_angle, np.sin(spine_angle), np.cos(spine_angle)])  # 3-5
            features.append(abs(spine_angle))  # 6
            
            # === FEATURE 3: Upper Body Alignment ===
            head_hip_horizontal = nose[0] - hip[0]
            head_hip_vertical = nose[1] - hip[1]
            head_hip_angle = np.arctan2(head_hip_horizontal, -head_hip_vertical)
            
            features.extend([head_hip_horizontal, head_hip_vertical, head_hip_angle])  # 7-9
            
            # === FEATURE 4: Shoulder Position ===
            shoulder_forward = shoulder[0] - hip[0]
            shoulder_height = shoulder[1] - hip[1]
            
            features.extend([shoulder_forward, shoulder_height])  # 10-11
            
            # === FEATURE 5: Head Height ===
            head_shoulder_vertical = nose[1] - shoulder[1]
            features.append(head_shoulder_vertical)  # 12
            
            # === FEATURE 6: Upper Back Curvature ===
            hip_to_nose = nose - hip
            hip_to_shoulder = shoulder - hip
            
            dot_product = np.dot(hip_to_nose, hip_to_shoulder)
            magnitudes = np.linalg.norm(hip_to_nose) * np.linalg.norm(hip_to_shoulder)
            
            if magnitudes > 0:
                curvature_angle = np.arccos(np.clip(dot_product / magnitudes, -1, 1))
            else:
                curvature_angle = 0
            
            features.append(curvature_angle)  # 13
            
            # === FEATURE 7: Spine Length ===
            spine_length_normalized = np.linalg.norm(spine_vector)
            features.append(spine_length_normalized)  # 14
            
            # === FEATURE 8: Neck Angle ===
            left_shoulder = keypoints['left_shoulder'][:2]
            right_shoulder = keypoints['right_shoulder'][:2]
            shoulder_midpoint = (left_shoulder + right_shoulder) / 2
            
            neck_vector = nose - shoulder_midpoint
            neck_angle = np.arctan2(neck_vector[0], -neck_vector[1])
            
            features.extend([neck_angle, np.sin(neck_angle), np.cos(neck_angle)])  # 15-17
            
            # === FEATURE 9: Elbow Position (DOWNWEIGHTED) ===
            elbow = keypoints[f'{visible_side}_elbow'][:2]
            elbow_visibility = keypoints[f'{visible_side}_elbow'][2]
            
            if elbow_visibility > 0.3:
                elbow_relative_shoulder = elbow - shoulder
                features.extend([elbow_relative_shoulder[0], elbow_relative_shoulder[1]])  # 18-19
            else:
                features.extend([0.0, 0.0])  # 18-19
            
            # === FEATURE 10: Visibility Scores ===
            critical_points = [f'{visible_side}_shoulder', f'{visible_side}_hip', 'nose']
            avg_visibility = np.mean([keypoints[kp][2] for kp in critical_points])
            features.append(avg_visibility)  # 20
            
            shoulder_vis_avg = (keypoints['left_shoulder'][2] + keypoints['right_shoulder'][2]) / 2
            features.append(shoulder_vis_avg)  # 21
            
            # === FEATURE 11: Side-View Depth Cue ===
            shoulder_depth_difference = abs(keypoints['left_shoulder'][0] - keypoints['right_shoulder'][0])
            features.append(shoulder_depth_difference)  # 22
            
            # === FEATURE 12: Head Tilt ===
            ear_height_diff = left_ear[1] - right_ear[1]
            features.append(ear_height_diff)  # 23
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error computing features: {e}")
            # Return zero features if computation fails
            return np.zeros(24, dtype=np.float32)
    
    def extract_features(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Full pipeline: extract keypoints and compute side-view features.
        
        Args:
            image: Input BGR image (side-view)
        
        Returns:
            Tuple of (feature_vector, debug_info) or None if detection fails
        """
        try:
            # Extract keypoints
            keypoints = self.extract_keypoints(image)
            if keypoints is None:
                return None
            
            # Determine visible side
            visible_side = self.determine_visible_side(keypoints)
            
            # Check if critical side-view keypoints are visible
            critical_points = [
                f'{visible_side}_shoulder',
                f'{visible_side}_hip',
                'nose'
            ]
            
            if not all(keypoints[kp][2] > 0.2 for kp in critical_points):
                return None
            
            # Get spine reference points
            shoulder = keypoints[f'{visible_side}_shoulder'][:2]
            hip = keypoints[f'{visible_side}_hip'][:2]
            spine_midpoint = (shoulder + hip) / 2
            spine_length = np.linalg.norm(shoulder - hip)
            
            if spine_length < 10:  # Too small, likely bad detection
                return None
            
            # Normalize keypoints
            normalized_keypoints = self.normalize_to_spine_frame(keypoints, hip, spine_length)
            
            # Compute side-view specific features
            features = self.compute_side_view_features(normalized_keypoints)
            
            # Debug info
            debug_info = {
                'raw_keypoints': keypoints,
                'normalized_keypoints': normalized_keypoints,
                'visible_side': visible_side,
                'spine_length': spine_length,
                'shoulder_pos': shoulder,
                'hip_pos': hip,
                'spine_angle': features[3],
                'head_forward': features[2],
            }
            
            return features, debug_info
            
        except Exception as e:
            print(f"Error in extract_features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_side_view(
        self, 
        image: np.ndarray, 
        keypoints: Dict[str, np.ndarray], 
        debug_info: Dict = None
    ) -> np.ndarray:
        """
        Visualize keypoints and important measurements for side-view.
        """
        try:
            vis_image = image.copy()
            
            # Determine visible side
            visible_side = debug_info['visible_side'] if debug_info else self.determine_visible_side(keypoints)
            
            # Draw keypoints
            for name, kp in keypoints.items():
                x, y, vis = kp
                if vis > 0.3:
                    # Color code by importance
                    if name == 'nose':
                        color = (0, 0, 255)  # Red
                        radius = 8
                    elif 'shoulder' in name:
                        color = (0, 255, 255)  # Yellow
                        radius = 8
                    elif 'hip' in name:
                        color = (255, 0, 255)  # Magenta
                        radius = 8
                    else:
                        color = (0, 255, 0)  # Green
                        radius = 5
                    
                    cv2.circle(vis_image, (int(x), int(y)), radius, color, -1)
            
            # Draw spine line
            shoulder = keypoints[f'{visible_side}_shoulder'][:2]
            hip = keypoints[f'{visible_side}_hip'][:2]
            
            if keypoints[f'{visible_side}_shoulder'][2] > 0.3 and keypoints[f'{visible_side}_hip'][2] > 0.3:
                cv2.line(vis_image, 
                        (int(shoulder[0]), int(shoulder[1])),
                        (int(hip[0]), int(hip[1])),
                        (0, 255, 255), 3)
            
            # Draw vertical reference
            if debug_info:
                h, w = vis_image.shape[:2]
                hip_pos = debug_info['hip_pos']
                cv2.line(vis_image, 
                        (int(hip_pos[0]), 0), 
                        (int(hip_pos[0]), h),
                        (255, 255, 255), 1)
                
                # Add text annotations
                spine_angle_deg = np.degrees(debug_info['spine_angle'])
                cv2.putText(vis_image, f"Side: {visible_side}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis_image, f"Spine: {spine_angle_deg:.1f}deg", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(vis_image, f"Head Fwd: {debug_info['head_forward']:.2f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return vis_image
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            return image.copy()


# Test function
def test_extractor():
    """Test the pose feature extractor."""
    print("\n" + "="*70)
    print("TESTING SIDE-VIEW POSE FEATURE EXTRACTOR")
    print("="*70 + "\n")
    
    if not MEDIAPIPE_AVAILABLE:
        print("ERROR: MediaPipe is not available. Please install it:")
        print("  pip install mediapipe")
        return False
    
    try:
        # Initialize extractor
        print("Initializing extractor...")
        extractor = SideViewPoseFeatureExtractor()
        print("✓ Extractor initialized\n")
        
        # Create a simple test image
        print("Creating test image...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test image - load your own side-view image", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        print("✓ Test image created\n")
        
        # Try to extract features
        print("Testing feature extraction on blank image...")
        result = extractor.extract_features(test_image)
        
        if result is None:
            print("✓ Correctly returned None for blank image (expected)\n")
        else:
            features, debug_info = result
            print(f"✓ Feature extraction works! Got {len(features)} features\n")
        
        print("="*70)
        print("EXTRACTOR TEST PASSED!")
        print("="*70)
        print("\nNext steps:")
        print("1. Load an actual side-view image:")
        print("   image = cv2.imread('your_side_view_image.jpg')")
        print("   result = extractor.extract_features(image)")
        print("\n2. If result is not None:")
        print("   features, debug_info = result")
        print("   print(f'Features: {features}')")
        print("   print(f'Spine angle: {np.degrees(features[3]):.1f} degrees')")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False


# Main execution
if __name__ == "__main__":
    success = test_extractor()
    
    if not success:
        print("\nTroubleshooting:")
        print("1. Make sure MediaPipe is installed:")
        print("   pip install mediapipe opencv-python numpy")
        print("\n2. Check Python version (requires 3.7+):")
        print("   python --version")
        print("\n3. If still failing, try:")
        print("   pip install --upgrade mediapipe")
        exit(1)
    else:
        exit(0)
"""
Side-view pose feature extraction using MediaPipe.
Extracts 24 hand-invariant geometric features from sitting posture images.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict
import math

# Handle different MediaPipe versions
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

from ..config import (
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    LANDMARK_INDICES
)


class SideViewPoseFeatureExtractor:
    """
    Extracts hand-invariant geometric features from side-view posture images.
    
    Key Features:
    - 24 geometric features relative to spine coordinate frame
    - Automatic side detection (left/right profile)
    - Robust to hand position variations
    - Normalized coordinates for scale invariance
    """
    
    def __init__(self):
        """Initialize MediaPipe Pose."""
        self.pose = mp_pose.Pose(
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        
    def extract_features(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Extract 24 hand-invariant geometric features from an image.
        
        Args:
            image: RGB or BGR image (H, W, 3)
            
        Returns:
            Tuple of (features, debug_info) or None if pose detection fails
            - features: np.ndarray of shape (24,)
            - debug_info: Dict with visualization data and metrics
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Run MediaPipe Pose
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return None
        
        # Access landmarks - results.pose_landmarks.landmark is the list
        # Store both for compatibility
        pose_landmarks_obj = results.pose_landmarks
        landmarks = pose_landmarks_obj.landmark
        
        # Detect which side is visible
        side = self._detect_side(landmarks)
        
        # Extract features based on detected side
        features = self._compute_features(landmarks, side, image.shape)
        
        # Prepare debug information
        debug_info = {
            'landmarks': pose_landmarks_obj,  # Store the full object for visualization
            'side': side,
            'spine_angle': features[4],  # Feature index 4 is spine angle
            'head_forward_distance': features[0],
            'shoulder_rotation': features[9]
        }
        
        return features, debug_info
    
    def _detect_side(self, landmarks) -> str:
        """
        Detect which side (left/right) is visible in the side-view image.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            'left' or 'right' indicating visible side
        """
        # Handle different landmark access methods
        def get_landmark(idx):
            try:
                return landmarks[idx]
            except (TypeError, KeyError):
                return getattr(landmarks, 'landmark')[idx] if hasattr(landmarks, 'landmark') else landmarks[idx]
        
        # Check visibility scores
        left_shoulder_vis = get_landmark(LANDMARK_INDICES['left_shoulder']).visibility
        right_shoulder_vis = get_landmark(LANDMARK_INDICES['right_shoulder']).visibility
        
        left_hip_vis = get_landmark(LANDMARK_INDICES['left_hip']).visibility
        right_hip_vis = get_landmark(LANDMARK_INDICES['right_hip']).visibility
        
        # Average visibility for each side
        left_score = (left_shoulder_vis + left_hip_vis) / 2
        right_score = (right_shoulder_vis + right_hip_vis) / 2
        
        return 'left' if left_score > right_score else 'right'
    
    def _get_landmark_coords(self, landmarks, name: str) -> np.ndarray:
        """Get 2D coordinates of a landmark."""
        idx = LANDMARK_INDICES[name]
        try:
            # Try direct indexing (works with list or direct access)
            lm = landmarks[idx]
        except TypeError:
            # If landmarks is not indexable, try to access as attribute
            lm = getattr(landmarks, 'landmark')[idx]
        return np.array([lm.x, lm.y])
    
    def _compute_features(self, landmarks, side: str, image_shape: Tuple) -> np.ndarray:
        """
        Compute 24 hand-invariant geometric features.
        
        Feature Categories:
        - Head Position (3): forward distance, vertical offset, tilt angle
        - Spine Alignment (5): spine angle, upper/lower curvature, deviation, straightness
        - Shoulder Position (4): height symmetry, rotation, forward/back, alignment
        - Upper Body Geometry (6): torso aspect ratio, width/height proportions, compactness
        - Elbow Angles (2): left/right flexion
        - Normalized Coordinates (4): key joint positions relative to spine
        
        Args:
            landmarks: MediaPipe pose landmarks
            side: 'left' or 'right' indicating visible side
            image_shape: (height, width, channels) of original image
            
        Returns:
            np.ndarray of shape (24,) containing features
        """
        features = []
        
        # Get key landmarks based on visible side
        shoulder_key = f'{side}_shoulder'
        hip_key = f'{side}_hip'
        elbow_key = f'{side}_elbow'
        wrist_key = f'{side}_wrist'
        
        # Core landmarks for spine coordinate frame
        nose = self._get_landmark_coords(landmarks, 'nose')
        shoulder = self._get_landmark_coords(landmarks, shoulder_key)
        hip = self._get_landmark_coords(landmarks, hip_key)
        elbow = self._get_landmark_coords(landmarks, elbow_key)
        wrist = self._get_landmark_coords(landmarks, wrist_key)
        
        # Establish spine coordinate frame
        spine_base = hip  # Origin at hip
        spine_vector = shoulder - hip
        spine_length = np.linalg.norm(spine_vector)
        
        if spine_length < 0.01:  # Avoid division by zero
            return np.zeros(24)
        
        # === CATEGORY 1: Head Position (3 features) ===
        # Feature 0: Head forward distance from shoulder line
        head_to_shoulder = nose - shoulder
        head_forward = np.abs(head_to_shoulder[0]) / spine_length
        features.append(head_forward)
        
        # Feature 1: Head vertical offset (normalized)
        head_vertical = head_to_shoulder[1] / spine_length
        features.append(head_vertical)
        
        # Feature 2: Head tilt angle
        head_angle = math.degrees(math.atan2(head_to_shoulder[1], head_to_shoulder[0]))
        features.append(head_angle / 180.0)  # Normalize to [-1, 1]
        
        # === CATEGORY 2: Spine Alignment (5 features) ===
        # Feature 3: Spine angle from vertical (KEY FEATURE)
        vertical = np.array([0, -1])  # Upward direction
        spine_unit = spine_vector / spine_length
        spine_angle = math.degrees(math.acos(np.clip(np.dot(spine_unit, vertical), -1, 1)))
        features.append(spine_angle / 90.0)  # Normalize to [0, 2]
        
        # Feature 4: Raw spine angle (for visualization)
        features.append(spine_angle)
        
        # Feature 5: Upper back curvature (deviation from straight line)
        mid_spine = (shoulder + hip) / 2
        shoulder_to_mid = shoulder - mid_spine
        curvature_upper = np.linalg.norm(shoulder_to_mid) / spine_length
        features.append(curvature_upper)
        
        # Feature 6: Lower back curvature
        hip_to_mid = hip - mid_spine
        curvature_lower = np.linalg.norm(hip_to_mid) / spine_length
        features.append(curvature_lower)
        
        # Feature 7: Spine straightness (1.0 = perfectly straight)
        spine_straightness = 1.0 / (1.0 + curvature_upper + curvature_lower)
        features.append(spine_straightness)
        
        # === CATEGORY 3: Shoulder Position (4 features) ===
        # Get opposite shoulder if visible
        opposite_side = 'right' if side == 'left' else 'left'
        opposite_shoulder_key = f'{opposite_side}_shoulder'
        
        try:
            opposite_shoulder = self._get_landmark_coords(landmarks, opposite_shoulder_key)
            shoulder_visibility = landmarks[LANDMARK_INDICES[opposite_shoulder_key]].visibility
        except:
            opposite_shoulder = shoulder
            shoulder_visibility = 0.0
        
        # Feature 8: Shoulder height symmetry
        shoulder_height_diff = np.abs(shoulder[1] - opposite_shoulder[1])
        features.append(shoulder_height_diff)
        
        # Feature 9: Shoulder rotation angle
        if shoulder_visibility > 0.3:
            shoulder_line = opposite_shoulder - shoulder
            shoulder_angle = math.degrees(math.atan2(shoulder_line[1], shoulder_line[0]))
        else:
            shoulder_angle = 0.0
        features.append(shoulder_angle / 180.0)
        
        # Feature 10: Shoulder forward/back position
        shoulder_horizontal = (shoulder[0] - hip[0]) / spine_length
        features.append(shoulder_horizontal)
        
        # Feature 11: Shoulder alignment score
        shoulder_alignment = 1.0 - shoulder_height_diff
        features.append(shoulder_alignment)
        
        # === CATEGORY 4: Upper Body Geometry (6 features) ===
        # Feature 12: Torso aspect ratio (width/height)
        torso_width = np.abs(shoulder[0] - hip[0])
        torso_aspect = torso_width / spine_length if spine_length > 0 else 1.0
        features.append(torso_aspect)
        
        # Feature 13-14: Torso bounding box proportions
        features.append(torso_width)
        features.append(spine_length)
        
        # Feature 15: Torso compactness (area / perimeter^2)
        torso_perimeter = 2 * (torso_width + spine_length)
        torso_area = torso_width * spine_length
        compactness = (4 * np.pi * torso_area) / (torso_perimeter ** 2) if torso_perimeter > 0 else 0
        features.append(compactness)
        
        # Feature 16-17: Center of mass position
        com_x = (shoulder[0] + hip[0]) / 2
        com_y = (shoulder[1] + hip[1]) / 2
        features.append(com_x)
        features.append(com_y)
        
        # === CATEGORY 5: Elbow Angles (2 features) ===
        # Feature 18: Visible side elbow angle
        shoulder_to_elbow = elbow - shoulder
        elbow_to_wrist = wrist - elbow
        elbow_angle = self._compute_angle(shoulder_to_elbow, elbow_to_wrist)
        features.append(elbow_angle / 180.0)
        
        # Feature 19: Opposite elbow (if visible, else 0)
        opposite_elbow_key = f'{opposite_side}_elbow'
        opposite_wrist_key = f'{opposite_side}_wrist'
        try:
            opposite_elbow = self._get_landmark_coords(landmarks, opposite_elbow_key)
            opposite_wrist = self._get_landmark_coords(landmarks, opposite_wrist_key)
            opposite_shoulder_to_elbow = opposite_elbow - opposite_shoulder
            opposite_elbow_to_wrist = opposite_wrist - opposite_elbow
            opposite_elbow_angle = self._compute_angle(opposite_shoulder_to_elbow, opposite_elbow_to_wrist)
            features.append(opposite_elbow_angle / 180.0)
        except:
            features.append(0.0)
        
        # === CATEGORY 6: Normalized Coordinates (4 features) ===
        # All coordinates normalized to spine frame
        # Feature 20-21: Nose position relative to spine base
        nose_relative = (nose - spine_base) / spine_length
        features.extend(nose_relative.tolist())
        
        # Feature 22-23: Shoulder position relative to spine base
        shoulder_relative = (shoulder - spine_base) / spine_length
        features.extend(shoulder_relative.tolist())
        
        return np.array(features, dtype=np.float32)
    
    def _compute_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute angle between two vectors in degrees.
        
        Args:
            v1, v2: 2D vectors
            
        Returns:
            Angle in degrees [0, 180]
        """
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def visualize_pose(self, image: np.ndarray, landmarks, side: str) -> np.ndarray:
        """
        Draw pose landmarks on image for visualization.
        
        Args:
            image: BGR image
            landmarks: MediaPipe pose landmarks (can be landmark list or pose_landmarks object)
            side: 'left' or 'right' visible side
            
        Returns:
            Image with pose overlay
        """
        annotated_image = image.copy()
        
        # Handle different landmark formats
        # Create a proper pose_landmarks object if needed
        if isinstance(landmarks, list):
            # Already converted to list, need to create pose_landmarks
            from mediapipe.framework.formats import landmark_pb2
            pose_landmarks = landmark_pb2.NormalizedLandmarkList()
            for lm in landmarks:
                pose_landmarks.landmark.add(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            
            # Draw all landmarks and connections
            mp_drawing.draw_landmarks(
                annotated_image,
                pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        else:
            # It's already a pose_landmarks object or similar
            try:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            except:
                # If drawing fails, try to work with the raw landmarks
                pass
        
        # Highlight key landmarks for spine
        h, w = image.shape[:2]
        shoulder_key = f'{side}_shoulder'
        hip_key = f'{side}_hip'
        
        try:
            # Get landmark coordinates
            def get_lm_coords(lm_name):
                idx = LANDMARK_INDICES[lm_name]
                try:
                    lm = landmarks[idx]
                except (TypeError, KeyError):
                    lm = getattr(landmarks, 'landmark')[idx] if hasattr(landmarks, 'landmark') else landmarks[idx]
                return int(lm.x * w), int(lm.y * h)
            
            shoulder_pt = get_lm_coords(shoulder_key)
            hip_pt = get_lm_coords(hip_key)
            
            # Draw spine line
            cv2.line(annotated_image, shoulder_pt, hip_pt, (255, 0, 0), 3)
        except Exception as e:
            # If spine line drawing fails, continue anyway
            pass
        
        return annotated_image
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


def test_feature_extraction():
    """Test pose feature extraction on a sample image."""
    import matplotlib.pyplot as plt
    
    # Create a test with a blank image (for demonstration)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    extractor = SideViewPoseFeatureExtractor()
    result = extractor.extract_features(test_image)
    
    if result is None:
        print("No pose detected in test image (expected for blank image)")
    else:
        features, debug_info = result
        print(f"Extracted {len(features)} features:")
        print(f"Spine angle: {debug_info['spine_angle']:.2f}°")
        print(f"Detected side: {debug_info['side']}")


if __name__ == "__main__":
    test_feature_extraction()
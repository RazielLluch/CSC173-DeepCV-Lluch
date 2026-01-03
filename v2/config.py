"""
Configuration file for Gamer ErgoVision project.
Contains all hyperparameters, paths, and constant definitions.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Posture Classes
POSTURE_CLASSES = [
    'backwardbadposture',
    'forwardbadposture',
    'goodposture'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(POSTURE_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# Risk Level Mapping
POSTURE_RISK_MAPPING = {
    'goodposture': 'Low Risk',
    'forwardbadposture': 'High Risk',
    'backwardbadposture': 'Medium Risk'
}

# Dataset Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Image Properties
IMAGE_SIZE = 224
INPUT_CHANNELS = 3

# Pose Feature Configuration
POSE_FEATURE_DIM = 24  # Number of geometric features

# Model Architecture Dimensions
POSE_MLP_HIDDEN_DIMS = [128, 256, 256]  # Pose branch MLP layers
APPEARANCE_FEATURE_DIM = 512  # ResNet18 output dimension
FUSED_FEATURE_DIM = 768  # 256 (pose) + 512 (appearance)
CLASSIFIER_HIDDEN_DIM = 256
NUM_CLASSES = 3

# Spatial Attention Configuration
ATTENTION_REDUCTION_RATIO = 4  # 512 -> 128 channels
ATTENTION_TOP_WEIGHT = 1.0
ATTENTION_BOTTOM_WEIGHT = 0.3

# Dropout Rates
POSE_MLP_DROPOUT = [0.3, 0.3, 0.2]  # For each layer
CLASSIFIER_DROPOUT = [0.3, 0.2]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-4

# Loss Weights
CLASSIFICATION_LOSS_WEIGHT = 1.0
CONSISTENCY_LOSS_WEIGHT = 0.3
CONSISTENCY_TEMPERATURE = 2.0

# Learning Rate Scheduler
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-6

# Early Stopping
EARLY_STOPPING_PATIENCE = 10

# Data Loading
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
PIN_MEMORY = True

# ============================================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================================
# Standard Augmentations
RANDOM_ROTATION_DEGREES = 10
COLOR_JITTER_BRIGHTNESS = (0.8, 1.2)
COLOR_JITTER_CONTRAST = (0.8, 1.2)

# ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Hand-Invariance Augmentations
HAND_OCCLUSION_PROB = 0.3
HAND_OCCLUSION_HEIGHT = (0.3, 0.5)  # Occlude bottom 30-50%
VERTICAL_SHIFT_PROB = 0.3
VERTICAL_SHIFT_RANGE = 0.15  # ±15% of height

# Consistency Augmentation (for training)
CONSISTENCY_VERTICAL_SHIFT_RANGE = 0.1  # ±10% for consistency loss

# ============================================================================
# MEDIAPIPE CONFIGURATION
# ============================================================================
MEDIAPIPE_MODEL_COMPLEXITY = 1  # 0, 1, or 2 (higher = more accurate, slower)
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# Landmark Indices (MediaPipe Pose has 33 landmarks)
LANDMARK_INDICES = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32,
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for predictions
DEVICE = 'cuda'  # Will be set dynamically in code

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
VIZ_FIGURE_SIZE = (15, 5)
VIZ_DPI = 100
VIZ_POSE_COLOR = (0, 255, 0)  # Green
VIZ_POSE_THICKNESS = 2
VIZ_LANDMARK_RADIUS = 5

# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================
CHECKPOINT_SAVE_BEST = True
CHECKPOINT_SAVE_LAST = True
BEST_MODEL_FILENAME = "best_model.pth"
LAST_MODEL_FILENAME = "last_model.pth"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_INTERVAL = 10  # Log every N batches
PRINT_METRICS = True
SAVE_TRAINING_PLOTS = True

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
EXPERIMENT_NAME = "gamer_ergovision"
MODEL_TYPE = "hybrid"  # Options: 'hybrid', 'appearance_only', 'pose_only'
FUSION_TYPE = "concatenation"  # Options: 'concatenation', 'weighted_addition', 'cross_attention'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_device():
    """Get the available device (cuda or cpu)."""
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_save_path(epoch=None, best=False):
    """Get the path to save/load model checkpoint."""
    if best:
        return CHECKPOINT_DIR / BEST_MODEL_FILENAME
    elif epoch is not None:
        return CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
    else:
        return CHECKPOINT_DIR / LAST_MODEL_FILENAME

def print_config():
    """Print the current configuration."""
    print("=" * 80)
    print("GAMER ERGOVISION - CONFIGURATION")
    print("=" * 80)
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Fusion Type: {FUSION_TYPE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Pose Features: {POSE_FEATURE_DIM}")
    print(f"Classes: {', '.join(POSTURE_CLASSES)}")
    print(f"Device: {get_device()}")
    print("=" * 80)

if __name__ == "__main__":
    print_config()
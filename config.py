"""
config.py
Configuration file for Gamer ErgoVision project.
Updated for your specific dataset structure.
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# TODO: Update this to match your actual project location
PROJECT_ROOT = Path(r"C:\Users\Admin\Desktop\School\CSC173\final project\CSC173-DeepCV-Lluch")
YOLO_DATASET_PATH = PROJECT_ROOT / "yolo_dataset"

# Model save paths
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
POSTURE_CLASSIFIER_PATH = MODELS_ROOT / "posture_classifier.pth"

# For detection (if you add detection later)
DETECTION_MODEL_PATH = MODELS_ROOT / "yolov8_detector.pt"

# Create directories if they don't exist
for path in [MODELS_ROOT, RESULTS_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

# Your actual posture classes from the dataset
# Note: The YAML has leading spaces, we'll strip them
RAW_POSTURE_CLASSES = [
    ' backwardbadposture',
    ' forwardbadposture', 
    ' goodposture'
]

# Cleaned class names (strip spaces)
POSTURE_CLASSES = [cls.strip() for cls in RAW_POSTURE_CLASSES]

# Map to more readable names for display/reports
POSTURE_DISPLAY_NAMES = {
    'backwardbadposture': 'Reclined/Backward',
    'forwardbadposture': 'Forward Lean',
    'goodposture': 'Neutral/Good'
}

# For your project, you can also map to the original intended categories:
POSTURE_TO_CATEGORY = {
    'goodposture': 'neutral',
    'forwardbadposture': 'forward_lean',
    'backwardbadposture': 'reclined'
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Posture Classifier Training
CLASSIFIER_CONFIG = {
    "backbone": "resnet18",    # Options: resnet18, resnet34, mobilenet_v3_small
    "pretrained": True,        # Use ImageNet pretrained weights
    "epochs": 30,              # TODO: Tune based on convergence
    "batch_size": 128,          # TODO: Adjust based on GPU memory (RTX 4050: 32-64 should work)
    "img_size": 224,           # Input size for classifier
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "num_workers": 2,          # Dataloader workers (reduce to 2 if on Windows and having issues)
    "device": "cuda",          # "cuda" or "cpu"
}

# YOLOv8 Detection Training (for future screen detection component)
YOLO_CONFIG = {
    "model_size": "yolov8s",
    "epochs": 50,
    "batch_size": 32,
    "img_size": 640,
    "learning_rate": 0.01,
    "patience": 10,
    "workers": 2,
    "device": "0",
}

# ============================================================================
# ERGONOMIC RISK ASSESSMENT PARAMETERS
# ============================================================================

# Distance thresholds (for future implementation with screen detection)
DISTANCE_THRESHOLDS = {
    "too_close": 0.3,
    "acceptable": 0.7,
    "far": 0.7,
}

# Ergonomic risk mapping: (posture, distance_bin) -> risk_level
# Using your three classes
RISK_MAPPING = {
    # Good posture - low risk regardless of distance
    ("goodposture", "too_close"): "low",
    ("goodposture", "acceptable"): "low",
    ("goodposture", "far"): "low",
    
    # Forward lean - varies with distance
    ("forwardbadposture", "too_close"): "high",
    ("forwardbadposture", "acceptable"): "medium",
    ("forwardbadposture", "far"): "low",
    
    # Backward/reclined - generally lower risk but not ideal
    ("backwardbadposture", "too_close"): "low",
    ("backwardbadposture", "acceptable"): "medium",
    ("backwardbadposture", "far"): "medium",
}

# Simplified risk mapping based on posture only (for current implementation)
POSTURE_RISK_MAPPING = {
    "goodposture": "low",
    "forwardbadposture": "high",
    "backwardbadposture": "medium"
}

DEFAULT_RISK_LEVEL = "medium"

# ============================================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================================

AUGMENTATION_CONFIG = {
    "rotation_range": 10,           # Degrees
    "horizontal_flip": True,        # Mirror image
    "brightness_range": (0.8, 1.2),
    "contrast_range": (0.8, 1.2),
}

# ============================================================================
# VISUALIZATION
# ============================================================================

# Risk level colors for display (BGR format for OpenCV)
RISK_COLORS = {
    "low": (0, 255, 0),     # Green
    "medium": (0, 165, 255), # Orange
    "high": (0, 0, 255),     # Red
}

# Posture colors for visualization
POSTURE_COLORS = {
    "goodposture": (0, 255, 0),           # Green
    "forwardbadposture": (0, 0, 255),     # Red
    "backwardbadposture": (0, 165, 255),  # Orange
}
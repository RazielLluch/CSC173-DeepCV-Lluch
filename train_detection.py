"""
train_detection.py
Train YOLOv8 object detector for person and screen detection.
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import torch

from config import (
    YOLO_CONFIG, DETECTION_CLASSES, SCREEN_DATASET_PATH,
    DETECTION_MODEL_PATH, RESULTS_ROOT, USE_WANDB, WANDB_PROJECT
)
from datasets import YOLODatasetPreparation


def prepare_detection_dataset():
    """
    Prepare and validate YOLO detection dataset.
    Creates data.yaml file if it doesn't exist.
    
    TODO: Adapt this based on your actual dataset structure.
    You may need to:
    1. Merge posture and screen datasets
    2. Relabel classes to match DETECTION_CLASSES
    3. Split into train/val if not already done
    """
    dataset_path = SCREEN_DATASET_PATH
    
    # TODO: Update these paths based on your dataset structure
    yaml_path = dataset_path / "data.yaml"
    
    if not yaml_path.exists():
        print("Creating data.yaml for YOLO training...")
        
        # Check if dataset has proper structure
        train_images = dataset_path / "images" / "train"
        val_images = dataset_path / "images" / "val"
        
        if not train_images.exists() or not val_images.exists():
            print(f"Warning: Expected YOLO format dataset at {dataset_path}")
            print("Expected structure:")
            print("  dataset/")
            print("    images/train/")
            print("    images/val/")
            print("    labels/train/")
            print("    labels/val/")
            return None
        
        # Create data.yaml
        YOLODatasetPreparation.create_yolo_yaml(
            dataset_path=dataset_path,
            train_path="images/train",
            val_path="images/val",
            class_names=DETECTION_CLASSES,
            yaml_path=yaml_path
        )
    
    # Validate dataset
    print("\nValidating training dataset...")
    train_stats = YOLODatasetPreparation.validate_yolo_dataset(dataset_path, "train")
    print(f"Train images: {train_stats['num_images']}")
    print(f"Train labels: {train_stats['num_labels']}")
    print(f"Class distribution: {train_stats['class_distribution']}")
    
    if train_stats['images_without_labels']:
        print(f"Warning: {len(train_stats['images_without_labels'])} images without labels")
    
    print("\nValidating validation dataset...")
    val_stats = YOLODatasetPreparation.validate_yolo_dataset(dataset_path, "val")
    print(f"Val images: {val_stats['num_images']}")
    print(f"Val labels: {val_stats['num_labels']}")
    
    return yaml_path


def train_yolo_detector(data_yaml: Path, resume: bool = False):
    """
    Train YOLOv8 detector for person and screen detection.
    
    Args:
        data_yaml: Path to data.yaml file
        resume: Whether to resume training from last checkpoint
    """
    print(f"\n{'='*60}")
    print("Starting YOLOv8 Detection Training")
    print(f"{'='*60}\n")
    
    # Initialize model
    model_name = f"{YOLO_CONFIG['model_size']}.pt"
    
    if resume and DETECTION_MODEL_PATH.exists():
        print(f"Resuming training from {DETECTION_MODEL_PATH}")
        model = YOLO(str(DETECTION_MODEL_PATH))
    else:
        print(f"Initializing new {YOLO_CONFIG['model_size']} model")
        model = YOLO(model_name)  # Load pretrained weights
    
    # Set up training arguments
    train_args = {
        'data': str(data_yaml),
        'epochs': YOLO_CONFIG['epochs'],
        'batch': YOLO_CONFIG['batch_size'],
        'imgsz': YOLO_CONFIG['img_size'],
        'lr0': YOLO_CONFIG['learning_rate'],
        'patience': YOLO_CONFIG['patience'],
        'workers': YOLO_CONFIG['workers'],
        'device': YOLO_CONFIG['device'],
        'project': str(RESULTS_ROOT / 'detection_training'),
        'name': 'yolov8_gamer_detector',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'save': True,
        'save_period': -1,  # Save checkpoint every N epochs (-1 = only save last and best)
        'plots': True,  # Generate training plots
    }
    
    # Add W&B integration if enabled
    if USE_WANDB:
        train_args['project'] = WANDB_PROJECT
    
    print("\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("Training started...")
    print("="*60 + "\n")
    
    # Train the model
    results = model.train(**train_args)
    
    # Save the best model to our specified path
    print(f"\nTraining complete! Saving best model to {DETECTION_MODEL_PATH}")
    best_model_path = RESULTS_ROOT / 'detection_training' / 'yolov8_gamer_detector' / 'weights' / 'best.pt'
    
    if best_model_path.exists():
        import shutil
        DETECTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model_path, DETECTION_MODEL_PATH)
        print(f"Model saved successfully!")
    
    return results


def evaluate_detector(data_yaml: Path):
    """
    Evaluate trained YOLOv8 detector on validation set.
    
    Args:
        data_yaml: Path to data.yaml file
    """
    if not DETECTION_MODEL_PATH.exists():
        print(f"Error: No trained model found at {DETECTION_MODEL_PATH}")
        print("Please train a model first using train_yolo_detector()")
        return None
    
    print(f"\n{'='*60}")
    print("Evaluating YOLOv8 Detector")
    print(f"{'='*60}\n")
    
    model = YOLO(str(DETECTION_MODEL_PATH))
    
    # Run validation
    results = model.val(
        data=str(data_yaml),
        batch=YOLO_CONFIG['batch_size'],
        imgsz=YOLO_CONFIG['img_size'],
        device=YOLO_CONFIG['device'],
        plots=True,
        save_json=True,
        project=str(RESULTS_ROOT / 'detection_evaluation'),
        name='val_results',
        exist_ok=True
    )
    
    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    
    # Per-class metrics
    print("\nPer-class AP50:")
    for i, class_name in enumerate(DETECTION_CLASSES):
        if i < len(results.box.ap50):
            print(f"  {class_name}: {results.box.ap50[i]:.4f}")
    
    return results


def test_inference(image_path: str):
    """
    Test inference on a single image.
    
    Args:
        image_path: Path to test image
    """
    if not DETECTION_MODEL_PATH.exists():
        print(f"Error: No trained model found at {DETECTION_MODEL_PATH}")
        return None
    
    print(f"\nRunning inference on {image_path}")
    
    model = YOLO(str(DETECTION_MODEL_PATH))
    
    # Run inference
    results = model.predict(
        source=image_path,
        imgsz=YOLO_CONFIG['img_size'],
        conf=0.25,
        save=True,
        project=str(RESULTS_ROOT / 'detection_inference'),
        name='test',
        exist_ok=True
    )
    
    # Print detections
    for r in results:
        print(f"\nDetections in {image_path}:")
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = DETECTION_CLASSES[class_id] if class_id < len(DETECTION_CLASSES) else "unknown"
            print(f"  {class_name}: {confidence:.2f}")
    
    print(f"\nResults saved to {RESULTS_ROOT / 'detection_inference' / 'test'}")
    
    return results


# Main training script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector for Gamer ErgoVision")
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['prepare', 'train', 'eval', 'test'],
                       help='Mode: prepare dataset, train, evaluate, or test')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--test-image', type=str, default=None,
                       help='Path to test image for inference')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    if args.mode == 'prepare':
        # Prepare and validate dataset
        yaml_path = prepare_detection_dataset()
        if yaml_path:
            print(f"\nDataset prepared! data.yaml at: {yaml_path}")
    
    elif args.mode == 'train':
        # Prepare dataset first
        yaml_path = prepare_detection_dataset()
        
        if yaml_path is None:
            print("Error: Could not prepare dataset. Please check dataset path and structure.")
        else:
            # Train model
            results = train_yolo_detector(yaml_path, resume=args.resume)
            
            # Automatically evaluate after training
            print("\n" + "="*60)
            print("Running post-training evaluation...")
            print("="*60)
            evaluate_detector(yaml_path)
    
    elif args.mode == 'eval':
        # Evaluate existing model
        yaml_path = SCREEN_DATASET_PATH / "data.yaml"
        if not yaml_path.exists():
            print("Error: data.yaml not found. Run with --mode prepare first.")
        else:
            evaluate_detector(yaml_path)
    
    elif args.mode == 'test':
        # Test inference on single image
        if args.test_image is None:
            print("Error: Please provide --test-image path")
        else:
            test_inference(args.test_image)
    
    print("\nDone!")
"""
datasets.py
Dataset loading utilities for your specific directory structure.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import yaml

from config import (
    POSTURE_CLASSES, CLASSIFIER_CONFIG, AUGMENTATION_CONFIG,
    YOLO_DATASET_PATH
)


class PostureDataset(Dataset):
    """
    PyTorch Dataset for your posture classification images.
    
    Expected structure:
        yolo_dataset/
            train/
                backwardbadposture/
                    img1.jpg
                    img2.jpg
                forwardbadposture/
                    img1.jpg
                goodposture/
                    img1.jpg
            val/
                ...
            test/
                ...
    """
    
    def __init__(
        self, 
        data_path: Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            data_path: Root path to yolo_dataset
            split: "train", "val", or "test"
            transform: Torchvision transforms to apply
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        
        # Load image paths and labels
        self.samples = []
        self._load_samples()
        
        # Create class-to-index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(POSTURE_CLASSES)}
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Classes: {POSTURE_CLASSES}")
        print(f"Class distribution:")
        self._print_class_distribution()
    
    def _load_samples(self):
        """Load image paths and labels from directory structure."""
        split_path = self.data_path / self.split
        
        print(split_path)

        if not split_path.exists():
            raise ValueError(f"Split path does not exist: {split_path}")
        
        # Use a set to track already-added files and prevent duplicates
        added_files = set()
        
        # Iterate through class folders
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            # Get class name and strip spaces (match your YAML format)
            class_name = class_dir.name.strip()
            
            if class_name not in POSTURE_CLASSES:
                print(f"Warning: Unknown class folder '{class_dir.name}', skipping...")
                continue
            
            # Load all images from this class (case-insensitive search)
            # Use a set to avoid duplicates within this class directory
            for img_path in class_dir.iterdir():
                if img_path.is_file():
                    # Check if it's an image file
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Use absolute path as unique identifier
                        file_key = str(img_path.resolve())
                        
                        if file_key not in added_files:
                            self.samples.append((img_path, class_name))
                            added_files.add(file_key)
    
    def _print_class_distribution(self):
        """Print distribution of classes in this split."""
        class_counts = {}
        for _, class_name in self.samples:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name in POSTURE_CLASSES:
            count = class_counts.get(class_name, 0)
            percentage = (count / len(self.samples) * 100) if self.samples else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    @property
    def image_paths(self):
        """Return list of image paths for debugging."""
        return [str(img_path) for img_path, _ in self.samples]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Transformed image tensor
            label: Integer class index
        """
        img_path, class_name = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get class index
        label = self.class_to_idx[class_name]
        
        return image, label


def get_posture_transforms(split: str = "train") -> transforms.Compose:
    """
    Get appropriate transforms for posture classification.
    
    Args:
        split: "train", "val", or "test"
    
    Returns:
        Composed transforms
    """
    img_size = CLASSIFIER_CONFIG["img_size"]
    
    if split == "train":
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(AUGMENTATION_CONFIG["rotation_range"]),
        ]
        
        if AUGMENTATION_CONFIG["horizontal_flip"]:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        transform_list.extend([
            transforms.ColorJitter(
                brightness=AUGMENTATION_CONFIG["brightness_range"],
                contrast=AUGMENTATION_CONFIG["contrast_range"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        transform = transforms.Compose(transform_list)
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def load_dataset_info():
    """Load and display dataset information from dataset.yaml"""
    yaml_path = YOLO_DATASET_PATH / "dataset.yaml"
    
    if not yaml_path.exists():
        print(f"Warning: dataset.yaml not found at {yaml_path}")
        return None
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print("\nDataset Information:")
    print(f"  Path: {data.get('path', 'N/A')}")
    print(f"  Train: {data.get('train', 'N/A')}")
    print(f"  Val: {data.get('val', 'N/A')}")
    print(f"  Test: {data.get('test', 'N/A')}")
    print(f"  Classes: {data.get('names', 'N/A')}")
    
    return data


# Example usage and validation
if __name__ == "__main__":
    print("="*60)
    print("Dataset Validation")
    print("="*60)
    
    # Load dataset info
    load_dataset_info()
    
    print("\n" + "="*60)
    print("Loading Training Set")
    print("="*60)
    
    # Load training dataset
    train_transform = get_posture_transforms("train")
    train_dataset = PostureDataset(
        YOLO_DATASET_PATH,
        split="train",
        transform=train_transform
    )
    
    print("\n" + "="*60)
    print("Loading Validation Set")
    print("="*60)
    
    # Load validation dataset
    val_transform = get_posture_transforms("val")
    val_dataset = PostureDataset(
        YOLO_DATASET_PATH,
        split="val",
        transform=val_transform
    )
    
    print("\n" + "="*60)
    print("Loading Test Set")
    print("="*60)
    
    # Load test dataset
    test_transform = get_posture_transforms("test")
    test_dataset = PostureDataset(
        YOLO_DATASET_PATH,
        split="test",
        transform=test_transform
    )
    
    # Show a sample
    if len(train_dataset) > 0:
        print("\n" + "="*60)
        print("Sample Data")
        print("="*60)
        img, label = train_dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Label: {label} ({POSTURE_CLASSES[label]})")
        print(f"Min pixel value: {img.min():.3f}")
        print(f"Max pixel value: {img.max():.3f}")
    
    print("\n✓ Dataset validation complete!")
"""
train_invariant.py
Training script with invariance-enforcing strategies.
Reduces sensitivity to absolute hand location.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2

from config import (
    POSTURE_CLASSES, CLASSIFIER_CONFIG, YOLO_DATASET_PATH,
    POSTURE_CLASSIFIER_PATH, RESULTS_ROOT, POSTURE_DISPLAY_NAMES
)
from pose_features import SideViewPoseFeatureExtractor
from hybrid_model import create_posture_model


class InvariantPostureDataset(torch.utils.data.Dataset):
    """
    Dataset that extracts both images and pose features.
    Applies spatial augmentations to enforce hand-invariance.
    """
    
    def __init__(
        self,
        data_path: Path,
        split: str,
        image_transform,
        use_pose_features: bool = True,
        apply_spatial_augmentation: bool = True,
        hand_occlusion_prob: float = 0.3
    ):
        from datasets import PostureDataset
        
        # Use existing dataset for image loading
        self.base_dataset = PostureDataset(data_path, split, transform=None)
        self.image_transform = image_transform
        self.use_pose_features = use_pose_features
        self.apply_spatial_augmentation = apply_spatial_augmentation and (split == "train")
        self.hand_occlusion_prob = hand_occlusion_prob
        
        # Initialize pose extractor
        if use_pose_features:
            from pose_features import SideViewPoseFeatureExtractor
            self.pose_extractor = SideViewPoseFeatureExtractor()
            print(f"  Pose extractor initialized for {split} set")
        
        # Precompute valid indices (images where pose can be extracted)
        if use_pose_features:
            print(f"Extracting pose features for {split} set...")
            self.valid_indices = []
            self.pose_features_cache = {}
            
            total = len(self.base_dataset)
            print(f"Processing {total} images...")
            
            for idx in range(total):
                # Print progress every 50 images
                if (idx + 1) % 50 == 0:
                    print(f"  Progress: {idx + 1}/{total} ({100*(idx+1)/total:.1f}%)")
                
                img_path, _ = self.base_dataset.samples[idx]
                img = cv2.imread(str(img_path))
                
                result = self.pose_extractor.extract_features(img)
                if result is not None:
                    features, _ = result
                    self.valid_indices.append(idx)
                    self.pose_features_cache[idx] = features
            
            print(f"  Completed: {total}/{total} (100.0%)")
            print(f"  Valid samples: {len(self.valid_indices)}/{len(self.base_dataset)}")
            
            if len(self.valid_indices) == 0:
                raise ValueError(
                    f"No valid poses found in {split} set! "
                    "Check that your images are side-view with visible person."
                )
        else:
            self.valid_indices = list(range(len(self.base_dataset)))
    
    def apply_hand_occlusion(self, image):
        """
        Randomly occlude hand/lower regions to force model to ignore them.
        This enforces hand-invariance during training.
        """
        if np.random.rand() > self.hand_occlusion_prob:
            return image
        
        h, w = image.shape[:2]
        
        # Occlude bottom 30-50% of image (where hands typically are)
        occlusion_start = int(h * np.random.uniform(0.5, 0.7))
        
        # Random occlusion type
        if np.random.rand() < 0.5:
            # Black rectangle
            image[occlusion_start:, :] = 0
        else:
            # Gaussian blur
            image[occlusion_start:, :] = cv2.GaussianBlur(
                image[occlusion_start:, :], (51, 51), 0
            )
        
        return image
    
    def apply_vertical_shift(self, image):
        """
        Randomly shift image vertically to change absolute position.
        Posture label should remain the same.
        """
        if np.random.rand() > 0.3:
            return image
        
        h, w = image.shape[:2]
        shift = int(h * np.random.uniform(-0.15, 0.15))
        
        M = np.float32([[1, 0, 0], [0, 1, shift]])
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return shifted
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get actual index
        actual_idx = self.valid_indices[idx]
        
        # Load image and label
        img_path, _ = self.base_dataset.samples[actual_idx]
        label = self.base_dataset.class_to_idx[self.base_dataset.samples[actual_idx][1]]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply spatial augmentations
        if self.apply_spatial_augmentation:
            image = self.apply_vertical_shift(image)
            image = self.apply_hand_occlusion(image)
        
        # Convert to PIL and apply standard transforms
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image)
        transformed_image = self.image_transform(pil_image)
        
        # Get pose features
        if self.use_pose_features:
            pose_features = self.pose_features_cache[actual_idx]
            pose_features = torch.from_numpy(pose_features)
            return transformed_image, pose_features, label
        else:
            return transformed_image, label


class ConsistencyLoss(nn.Module):
    """
    Consistency regularization: predictions should be stable under
    transformations that don't change posture.
    
    Creates two views of same image with different hand positions,
    then enforces prediction consistency.
    """
    
    def __init__(self, weight: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def create_augmented_view(self, images):
        """
        Create augmented view by shifting vertical position.
        This simulates hands being in different locations.
        """
        augmented = []
        for img in images:
            # Convert to numpy for cv2
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            
            # Random vertical shift
            h, w = img_np.shape[:2]
            shift = int(h * np.random.uniform(-0.1, 0.1))
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            shifted = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Convert back to tensor
            shifted_tensor = torch.from_numpy(shifted).permute(2, 0, 1).float() / 255.0
            augmented.append(shifted_tensor)
        
        return torch.stack(augmented).to(images.device)
    
    def forward(self, model, images, pose_features):
        """
        Compute consistency loss between original and augmented views.
        """
        # Original predictions
        with torch.no_grad():
            original_logits = model(images, pose_features)
            original_probs = torch.softmax(original_logits / self.temperature, dim=1)
        
        # Augmented predictions
        augmented_images = self.create_augmented_view(images)
        augmented_logits = model(augmented_images, pose_features)
        augmented_log_probs = torch.log_softmax(augmented_logits / self.temperature, dim=1)
        
        # KL divergence between original and augmented predictions
        consistency_loss = self.kl_div(augmented_log_probs, original_probs)
        
        return self.weight * consistency_loss


def train_epoch_with_invariance(
    model, dataloader, criterion, consistency_loss, optimizer, device, use_consistency: bool
):
    """Train for one epoch with invariance regularization."""
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_cons_loss = 0.0
    correct = 0
    total = 0
    
    print(f"Starting epoch with {len(dataloader)} batches...")
    
    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}")
        
        try:
            if len(batch) == 3:
                images, pose_features, labels = batch
                images = images.to(device)
                pose_features = pose_features.to(device)
                labels = labels.to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                pose_features = None
            
            # Forward pass
            optimizer.zero_grad()
            
            if pose_features is not None:
                outputs = model(images, pose_features)
            else:
                outputs = model(images)
            
            # Classification loss
            cls_loss = criterion(outputs, labels)
            
            # Consistency loss (enforce invariance)
            if use_consistency and pose_features is not None:
                cons_loss = consistency_loss(model, images, pose_features)
            else:
                cons_loss = torch.tensor(0.0).to(device)
            
            # Total loss
            total_loss = cls_loss + cons_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += total_loss.item() * images.size(0)
            running_cls_loss += cls_loss.item() * images.size(0)
            running_cons_loss += cons_loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            batch_count += 1
            
        except Exception as e:
            print(f"\n❌ Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if total == 0:
        print("❌ No samples processed in epoch!")
        return 0.0, 0.0, 0.0, 0.0
    
    epoch_loss = running_loss / total
    epoch_cls_loss = running_cls_loss / total
    epoch_cons_loss = running_cons_loss / total
    epoch_acc = 100. * correct / total
    
    print(f"Epoch complete: {batch_count} batches processed")
    
    return epoch_loss, epoch_cls_loss, epoch_cons_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    print(f"Validating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if (batch_idx + 1) % 5 == 0:
                print(f"  Val batch {batch_idx + 1}/{len(dataloader)}")
            
            if len(batch) == 3:
                images, pose_features, labels = batch
                images = images.to(device)
                pose_features = pose_features.to(device)
                labels = labels.to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                pose_features = None
            
            # Forward pass
            if pose_features is not None:
                outputs = model(images, pose_features)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    print(f"Validation complete")
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def main():
    """Main training function with invariance regularization."""
    print("\n" + "="*70)
    print(" " * 10 + "TRAINING WITH HAND-INVARIANCE REGULARIZATION")
    print("="*70 + "\n")
    
    # Create results directory
    results_dir = RESULTS_ROOT / "invariant_training"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Model type selection
    print("Model options:")
    print("  1. hybrid - Pose features + Appearance (recommended)")
    print("  2. pose_only - Only geometric features (most invariant)")
    print("  3. appearance_only - Original model (baseline)")
    
    model_type = "hybrid"  # Change this to test different architectures
    use_consistency = True  # Enable consistency regularization
    
    print(f"\nUsing: {model_type}")
    print(f"Consistency regularization: {use_consistency}\n")
    
    # Load datasets
    print("Loading datasets with pose features...")
    from datasets import get_posture_transforms
    
    train_transform = get_posture_transforms("train")
    val_transform = get_posture_transforms("val")
    
    use_pose = model_type in ["hybrid", "pose_only"]
    
    train_dataset = InvariantPostureDataset(
        YOLO_DATASET_PATH, "train", train_transform,
        use_pose_features=use_pose,
        apply_spatial_augmentation=True,
        hand_occlusion_prob=0.3
    )
    
    val_dataset = InvariantPostureDataset(
        YOLO_DATASET_PATH, "val", val_transform,
        use_pose_features=use_pose,
        apply_spatial_augmentation=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CLASSIFIER_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,  # MUST be 0 on Windows to avoid multiprocessing errors
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CLASSIFIER_CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,  # MUST be 0 on Windows to avoid multiprocessing errors
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataset loaded!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")
    
    # Initialize model
    print(f"Initializing {model_type} model...")
    
    if model_type == "hybrid":
        model = create_posture_model(
            num_classes=len(POSTURE_CLASSES),
            model_type="hybrid",
            pose_feature_dim=24,
            appearance_backbone="resnet18",
            fusion_method="concat"
        )
    elif model_type == "pose_only":
        model = create_posture_model(
            num_classes=len(POSTURE_CLASSES),
            model_type="pose_only",
            pose_feature_dim=24
        )
    else:
        model = create_posture_model(
            num_classes=len(POSTURE_CLASSES),
            model_type="appearance_only",
            backbone="resnet18"
        )
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    consistency_loss = ConsistencyLoss(weight=0.5, temperature=1.0)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=CLASSIFIER_CONFIG["learning_rate"],
        weight_decay=CLASSIFIER_CONFIG["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_cls_loss': [], 'train_cons_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(CLASSIFIER_CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CLASSIFIER_CONFIG['epochs']}")
        print("="*70)
        
        # Train
        train_loss, train_cls_loss, train_cons_loss, train_acc = train_epoch_with_invariance(
            model, train_loader, criterion, consistency_loss, optimizer, device, use_consistency
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_cons_loss'].append(train_cons_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}, cons: {train_cons_loss:.4f})")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = results_dir / f"best_model_{model_type}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_type': model_type,
            }, save_path)
            print(f"  ✓ Best model saved! ({save_path})")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with invariance regularization")
    parser.add_argument('--model', type=str, default='hybrid',
                       choices=['hybrid', 'pose_only', 'appearance_only'],
                       help='Model type')
    parser.add_argument('--no-consistency', action='store_true',
                       help='Disable consistency regularization')
    parser.add_argument('--epochs', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.epochs:
        CLASSIFIER_CONFIG["epochs"] = args.epochs
    
    main()
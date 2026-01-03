"""
Data augmentation transforms for hand-invariant posture classification.
Includes standard augmentations and hand-invariance specific transforms.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image
from typing import Tuple

from ..config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RANDOM_ROTATION_DEGREES,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    HAND_OCCLUSION_PROB,
    HAND_OCCLUSION_HEIGHT,
    VERTICAL_SHIFT_PROB,
    VERTICAL_SHIFT_RANGE,
    CONSISTENCY_VERTICAL_SHIFT_RANGE
)


class HandOcclusion:
    """
    Randomly occlude the bottom portion of the image to force
    the model to ignore hand regions.
    """
    
    def __init__(self, prob: float = HAND_OCCLUSION_PROB, 
                 height_range: Tuple[float, float] = HAND_OCCLUSION_HEIGHT):
        """
        Args:
            prob: Probability of applying occlusion
            height_range: (min, max) fraction of image height to occlude from bottom
        """
        self.prob = prob
        self.height_range = height_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply hand occlusion to PIL Image.
        
        Args:
            img: PIL Image
            
        Returns:
            PIL Image with possible occlusion
        """
        if random.random() > self.prob:
            return img
        
        # Convert to tensor for manipulation
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Random occlusion height
        occlusion_fraction = random.uniform(*self.height_range)
        occlusion_height = int(h * occlusion_fraction)
        occlusion_start = h - occlusion_height
        
        # Choose occlusion type: black rectangle or Gaussian blur
        if random.random() < 0.5:
            # Black rectangle
            img_array[occlusion_start:, :] = 0
        else:
            # Gaussian blur (simulate out-of-focus hands)
            blur_kernel = random.choice([15, 21, 31])
            import cv2
            img_array[occlusion_start:, :] = cv2.GaussianBlur(
                img_array[occlusion_start:, :],
                (blur_kernel, blur_kernel),
                0
            )
        
        return Image.fromarray(img_array)


class VerticalShift:
    """
    Randomly shift the image vertically to enforce position invariance.
    Changes absolute position without changing posture.
    """
    
    def __init__(self, prob: float = VERTICAL_SHIFT_PROB,
                 shift_range: float = VERTICAL_SHIFT_RANGE):
        """
        Args:
            prob: Probability of applying shift
            shift_range: Maximum shift as fraction of image height (±)
        """
        self.prob = prob
        self.shift_range = shift_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply vertical shift to PIL Image.
        
        Args:
            img: PIL Image
            
        Returns:
            PIL Image with possible shift
        """
        if random.random() > self.prob:
            return img
        
        # Random shift amount
        shift_fraction = random.uniform(-self.shift_range, self.shift_range)
        shift_pixels = int(img.height * shift_fraction)
        
        # Apply translation
        img = TF.affine(
            img,
            angle=0,
            translate=(0, shift_pixels),
            scale=1.0,
            shear=0,
            fill=0
        )
        
        return img


class ConsistencyVerticalShift:
    """
    Vertical shift specifically for consistency loss.
    Always applies a shift (no probability).
    """
    
    def __init__(self, shift_range: float = CONSISTENCY_VERTICAL_SHIFT_RANGE):
        """
        Args:
            shift_range: Maximum shift as fraction of image height (±)
        """
        self.shift_range = shift_range
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply vertical shift to tensor image.
        
        Args:
            img: Tensor of shape (C, H, W)
            
        Returns:
            Shifted tensor
        """
        shift_fraction = random.uniform(-self.shift_range, self.shift_range)
        shift_pixels = int(img.shape[1] * shift_fraction)
        
        # Use torch roll for efficient shifting
        if shift_pixels != 0:
            img = torch.roll(img, shifts=shift_pixels, dims=1)
            
            # Zero out rolled regions
            if shift_pixels > 0:
                img[:, :shift_pixels, :] = 0
            else:
                img[:, shift_pixels:, :] = 0
        
        return img


def get_posture_transforms(split: str = 'train'):
    """
    Get the appropriate transform pipeline for train/val/test.
    
    Args:
        split: One of 'train', 'val', 'test'
        
    Returns:
        torchvision.transforms.Compose object
    """
    if split == 'train':
        # Training: Standard augmentations + hand-invariance augmentations
        transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            
            # Standard augmentations
            T.RandomRotation(degrees=RANDOM_ROTATION_DEGREES),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=COLOR_JITTER_BRIGHTNESS,
                contrast=COLOR_JITTER_CONTRAST
            ),
            
            # Hand-invariance augmentations
            HandOcclusion(prob=HAND_OCCLUSION_PROB),
            VerticalShift(prob=VERTICAL_SHIFT_PROB),
            
            # Convert to tensor and normalize
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    else:
        # Validation/Test: Only resize and normalize (no augmentation)
        transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    return transform


def get_consistency_transform():
    """
    Get transform for consistency loss (vertical shift only).
    
    Returns:
        ConsistencyVerticalShift instance
    """
    return ConsistencyVerticalShift()


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W) or (B, C, H, W)
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    # Denormalize
    tensor = tensor * std + mean
    
    # Clip to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor


def visualize_augmentations():
    """
    Visualize the effect of augmentations on a sample image.
    """
    import matplotlib.pyplot as plt
    
    # Create a simple test image
    test_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='white')
    
    # Draw a simple stick figure
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    
    # Draw body
    draw.ellipse([100, 30, 124, 54], fill='black')  # Head
    draw.rectangle([110, 54, 114, 120], fill='black')  # Spine
    draw.rectangle([80, 60, 110, 64], fill='black')  # Left arm
    draw.rectangle([114, 60, 144, 64], fill='black')  # Right arm
    draw.rectangle([105, 120, 109, 180], fill='black')  # Left leg
    draw.rectangle([115, 120, 119, 180], fill='black')  # Right leg
    
    # Apply augmentations
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original
    axes[0, 0].imshow(test_image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Hand Occlusion
    occlusion = HandOcclusion(prob=1.0)
    occluded = occlusion(test_image.copy())
    axes[0, 1].imshow(occluded)
    axes[0, 1].set_title('Hand Occlusion')
    axes[0, 1].axis('off')
    
    # Vertical Shift
    shift = VerticalShift(prob=1.0)
    shifted = shift(test_image.copy())
    axes[0, 2].imshow(shifted)
    axes[0, 2].set_title('Vertical Shift')
    axes[0, 2].axis('off')
    
    # Combined augmentations
    combined = get_posture_transforms('train')
    for i in range(3):
        aug_image = test_image.copy()
        aug_tensor = combined(aug_image)
        aug_display = denormalize_image(aug_tensor).permute(1, 2, 0).numpy()
        axes[1, i].imshow(aug_display)
        axes[1, i].set_title(f'Full Pipeline {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved augmentation visualization to 'augmentation_visualization.png'")
    plt.close()


if __name__ == "__main__":
    print("Testing data augmentation pipeline...")
    visualize_augmentations()
    print("Done!")
"""
Hybrid hand-invariant posture classification model.
Combines geometric pose features with appearance features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple

from ..config import (
    POSE_FEATURE_DIM,
    POSE_MLP_HIDDEN_DIMS,
    POSE_MLP_DROPOUT,
    APPEARANCE_FEATURE_DIM,
    ATTENTION_REDUCTION_RATIO,
    ATTENTION_TOP_WEIGHT,
    ATTENTION_BOTTOM_WEIGHT,
    CLASSIFIER_HIDDEN_DIM,
    CLASSIFIER_DROPOUT,
    NUM_CLASSES,
    FUSION_TYPE
)


class SpatialAttentionMask(nn.Module):
    """
    Spatial attention module that downweights hand regions (bottom of image).
    
    Combines:
    1. Learned attention weights
    2. Position-based prior (top-to-bottom gradient)
    """
    
    def __init__(self, in_channels: int = APPEARANCE_FEATURE_DIM):
        """
        Args:
            in_channels: Number of input feature map channels
        """
        super().__init__()
        
        # Learnable attention
        reduced_channels = in_channels // ATTENTION_REDUCTION_RATIO
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Position-based prior parameters
        self.top_weight = ATTENTION_TOP_WEIGHT
        self.bottom_weight = ATTENTION_BOTTOM_WEIGHT
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to feature maps.
        
        Args:
            x: Feature maps of shape (B, C, H, W)
            
        Returns:
            Attended feature maps of shape (B, C, H, W)
        """
        batch_size, channels, height, width = x.shape
        
        # 1. Learn spatial attention weights
        attention_map = self.attention_conv(x)  # (B, 1, H, W)
        
        # 2. Create position-based prior (linear gradient from top to bottom)
        y_coords = torch.linspace(
            self.top_weight,
            self.bottom_weight,
            height,
            device=x.device
        )
        position_prior = y_coords.view(1, 1, height, 1)  # (1, 1, H, 1)
        
        # 3. Combine learned attention with position prior
        final_attention = attention_map * position_prior  # (B, 1, H, W)
        
        # 4. Apply attention to feature maps
        attended_features = x * final_attention
        
        return attended_features


class PoseFeatureMLP(nn.Module):
    """
    MLP for processing geometric pose features.
    Extracts high-level pose representations from 24 geometric features.
    """
    
    def __init__(
        self,
        input_dim: int = POSE_FEATURE_DIM,
        hidden_dims: list = POSE_MLP_HIDDEN_DIMS,
        dropout_rates: list = POSE_MLP_DROPOUT
    ):
        """
        Args:
            input_dim: Dimension of input pose features
            hidden_dims: List of hidden layer dimensions
            dropout_rates: List of dropout rates for each layer
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, dropout) in enumerate(zip(hidden_dims, dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            layers.append(nn.Dropout(p=dropout))
            
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process pose features.
        
        Args:
            x: Pose features of shape (B, POSE_FEATURE_DIM)
            
        Returns:
            Pose embeddings of shape (B, output_dim)
        """
        return self.mlp(x)


class AppearanceCNN(nn.Module):
    """
    Appearance feature extraction using ResNet18 backbone.
    Includes spatial attention to downweight hand regions.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load ResNet18 backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove final FC layer (we only want feature maps)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Spatial attention module
        self.spatial_attention = SpatialAttentionMask(in_channels=512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.output_dim = 512
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract appearance features.
        
        Args:
            x: Images of shape (B, 3, 224, 224)
            
        Returns:
            Appearance embeddings of shape (B, 512)
        """
        # Extract feature maps
        features = self.backbone(x)  # (B, 512, H, W)
        
        # Apply spatial attention
        attended_features = self.spatial_attention(features)  # (B, 512, H, W)
        
        # Global pooling
        pooled_features = self.global_pool(attended_features)  # (B, 512, 1, 1)
        
        # Flatten
        embeddings = pooled_features.flatten(1)  # (B, 512)
        
        return embeddings


class FusionLayer(nn.Module):
    """
    Fusion layer for combining pose and appearance features.
    Supports multiple fusion strategies.
    """
    
    def __init__(
        self,
        pose_dim: int,
        appearance_dim: int,
        fusion_type: str = FUSION_TYPE
    ):
        """
        Args:
            pose_dim: Dimension of pose embeddings
            appearance_dim: Dimension of appearance embeddings
            fusion_type: One of 'concatenation', 'weighted_addition', 'cross_attention'
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        self.pose_dim = pose_dim
        self.appearance_dim = appearance_dim
        
        if fusion_type == 'concatenation':
            # Simple concatenation
            self.output_dim = pose_dim + appearance_dim
            
        elif fusion_type == 'weighted_addition':
            # Project pose to same dim as appearance, then weighted sum
            self.pose_projection = nn.Linear(pose_dim, appearance_dim)
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            self.output_dim = appearance_dim
            
        elif fusion_type == 'cross_attention':
            # Cross-attention between pose and appearance
            embed_dim = min(pose_dim, appearance_dim)
            self.pose_projection = nn.Linear(pose_dim, embed_dim)
            self.appearance_projection = nn.Linear(appearance_dim, embed_dim)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=4,
                batch_first=True
            )
            self.output_dim = embed_dim
            
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        pose_features: torch.Tensor,
        appearance_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse pose and appearance features.
        
        Args:
            pose_features: (B, pose_dim)
            appearance_features: (B, appearance_dim)
            
        Returns:
            Fused features of shape (B, output_dim)
        """
        if self.fusion_type == 'concatenation':
            # Simple concatenation
            fused = torch.cat([pose_features, appearance_features], dim=1)
            
        elif self.fusion_type == 'weighted_addition':
            # Project and weighted sum
            pose_proj = self.pose_projection(pose_features)
            weight = torch.sigmoid(self.fusion_weight)
            fused = weight * pose_proj + (1 - weight) * appearance_features
            
        elif self.fusion_type == 'cross_attention':
            # Project both features
            pose_proj = self.pose_projection(pose_features).unsqueeze(1)  # (B, 1, D)
            app_proj = self.appearance_projection(appearance_features).unsqueeze(1)  # (B, 1, D)
            
            # Cross-attention (pose queries appearance)
            attended, _ = self.cross_attention(
                query=pose_proj,
                key=app_proj,
                value=app_proj
            )
            fused = attended.squeeze(1)  # (B, D)
        
        return fused


class ClassificationHead(nn.Module):
    """
    Classification head for final prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = CLASSIFIER_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout_rates: list = CLASSIFIER_DROPOUT
    ):
        """
        Args:
            input_dim: Dimension of fused features
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout_rates: Dropout rates for layers
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rates[0]),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rates[1]),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify fused features.
        
        Args:
            x: Fused features of shape (B, input_dim)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.classifier(x)


class HybridPostureClassifier(nn.Module):
    """
    Complete hybrid model for hand-invariant posture classification.
    
    Architecture:
    1. Pose Branch: MLP processes 24 geometric features
    2. Appearance Branch: ResNet18 + Spatial Attention
    3. Fusion Layer: Combines both branches
    4. Classification Head: Final prediction
    """
    
    def __init__(
        self,
        fusion_type: str = FUSION_TYPE,
        pretrained_backbone: bool = True
    ):
        """
        Args:
            fusion_type: Fusion strategy ('concatenation', 'weighted_addition', 'cross_attention')
            pretrained_backbone: Whether to use pretrained ResNet18
        """
        super().__init__()
        
        # Pose feature branch
        self.pose_branch = PoseFeatureMLP()
        
        # Appearance feature branch
        self.appearance_branch = AppearanceCNN(pretrained=pretrained_backbone)
        
        # Fusion layer
        self.fusion = FusionLayer(
            pose_dim=self.pose_branch.output_dim,
            appearance_dim=self.appearance_branch.output_dim,
            fusion_type=fusion_type
        )
        
        # Classification head
        self.classifier = ClassificationHead(input_dim=self.fusion.output_dim)
        
        self.fusion_type = fusion_type
        
    def forward(
        self,
        images: torch.Tensor,
        pose_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            images: Images of shape (B, 3, 224, 224)
            pose_features: Pose features of shape (B, 24)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        # Extract features from both branches
        pose_embeddings = self.pose_branch(pose_features)
        appearance_embeddings = self.appearance_branch(images)
        
        # Fuse features
        fused_features = self.fusion(pose_embeddings, appearance_embeddings)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_feature_embeddings(
        self,
        images: torch.Tensor,
        pose_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate feature embeddings for analysis.
        
        Args:
            images: Images of shape (B, 3, 224, 224)
            pose_features: Pose features of shape (B, 24)
            
        Returns:
            Tuple of (pose_embeddings, appearance_embeddings, fused_features)
        """
        with torch.no_grad():
            pose_embeddings = self.pose_branch(pose_features)
            appearance_embeddings = self.appearance_branch(images)
            fused_features = self.fusion(pose_embeddings, appearance_embeddings)
        
        return pose_embeddings, appearance_embeddings, fused_features


def create_model(
    fusion_type: str = FUSION_TYPE,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> HybridPostureClassifier:
    """
    Create and initialize the hybrid model.
    
    Args:
        fusion_type: Fusion strategy
        pretrained: Whether to use pretrained backbone
        device: Device to move model to
        
    Returns:
        Initialized HybridPostureClassifier
    """
    model = HybridPostureClassifier(
        fusion_type=fusion_type,
        pretrained_backbone=pretrained
    )
    
    if device is not None:
        model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    print(f"Fusion Type: {fusion_type}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print("="*80)
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)
    
    # Test forward pass
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_pose = torch.randn(batch_size, 24).to(device)
    
    with torch.no_grad():
        logits = model(dummy_images, dummy_pose)
    
    print(f"\nTest forward pass:")
    print(f"Input images shape: {dummy_images.shape}")
    print(f"Input pose shape: {dummy_pose.shape}")
    print(f"Output logits shape: {logits.shape}")
    print("\nModel test successful!")
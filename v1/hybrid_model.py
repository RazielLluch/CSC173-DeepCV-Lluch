"""
hybrid_model.py
Hybrid architecture combining pose features and appearance features.
Reduces dependence on absolute hand location while preserving visual cues.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Tuple


class SpatialAttentionMask(nn.Module):
    """
    Spatial attention that downweights hand/lap regions.
    Forces model to focus on upper body.
    """
    
    def __init__(self, feature_channels: int):
        super().__init__()
        
        # Learn spatial attention weights
        self.attention = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(feature_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        attention_map = self.attention(x)  # [B, 1, H, W]
        
        # Downweight bottom region (where hands usually are)
        B, _, H, W = attention_map.shape
        
        # Create position-based prior: weight decreases toward bottom
        y_coords = torch.linspace(1.0, 0.3, H, device=x.device).view(1, 1, H, 1)
        position_prior = y_coords.expand(B, 1, H, W)
        
        # Combine learned attention with position prior
        final_attention = attention_map * position_prior
        
        # Apply attention
        return x * final_attention


class PoseFeatureMLP(nn.Module):
    """
    Process geometric pose features through MLP.
    These features are already hand-invariant by design.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class AppearanceCNN(nn.Module):
    """
    CNN for processing cropped upper body image.
    Uses spatial attention to focus on torso/shoulders.
    """
    
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
            feature_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            feature_dim = 512
        elif backbone == "mobilenet_v3_small":
            mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
            self.features = mobilenet.features
            feature_dim = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.feature_dim = feature_dim
        
        # Add spatial attention to focus on upper body
        self.spatial_attention = SpatialAttentionMask(feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # Extract features
        features = self.features(x)  # [B, C, H, W]
        
        # Apply spatial attention (downweight bottom regions)
        attended_features = self.spatial_attention(features)
        
        # Global pool
        pooled = self.global_pool(attended_features)  # [B, C, 1, 1]
        pooled = pooled.flatten(1)  # [B, C]
        
        return pooled


class HybridPostureClassifier(nn.Module):
    """
    Hybrid model combining:
    1. Geometric pose features (hand-invariant by design, optimized for side-view)
    2. Appearance features (with spatial attention to focus on upper body profile)
    
    This architecture reduces dependence on absolute hand location while
    preserving rich visual information about posture from side-view camera angles.
    """
    
    def __init__(
        self,
        num_classes: int,
        pose_feature_dim: int = 24,  # From SideViewPoseFeatureExtractor
        appearance_backbone: str = "resnet18",
        pretrained: bool = True,
        fusion_method: str = "concat",  # "concat", "add", or "attention"
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # === Pose Feature Branch ===
        self.pose_mlp = PoseFeatureMLP(
            input_dim=pose_feature_dim,
            hidden_dims=[128, 256, 256],
            dropout=0.3
        )
        pose_output_dim = 256
        
        # === Appearance Branch ===
        self.appearance_cnn = AppearanceCNN(
            backbone=appearance_backbone,
            pretrained=pretrained
        )
        appearance_output_dim = self.appearance_cnn.feature_dim
        
        # === Fusion ===
        if fusion_method == "concat":
            # Simple concatenation
            fusion_dim = pose_output_dim + appearance_output_dim
            self.fusion = None
        
        elif fusion_method == "add":
            # Weighted addition (requires same dimensions)
            self.pose_projection = nn.Linear(pose_output_dim, appearance_output_dim)
            self.fusion_weight = nn.Parameter(torch.tensor([0.5]))  # Learnable weight
            fusion_dim = appearance_output_dim
        
        elif fusion_method == "attention":
            # Cross-attention fusion
            self.fusion = nn.MultiheadAttention(
                embed_dim=256,
                num_heads=4,
                batch_first=True
            )
            self.pose_projection = nn.Linear(pose_output_dim, 256)
            self.appearance_projection = nn.Linear(appearance_output_dim, 256)
            fusion_dim = 256
        
        # === Classification Head ===
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(
        self, 
        image: torch.Tensor, 
        pose_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: [B, 3, H, W] - RGB image
            pose_features: [B, pose_feature_dim] - Geometric pose features
        
        Returns:
            logits: [B, num_classes]
        """
        # Process pose features
        pose_embedding = self.pose_mlp(pose_features)  # [B, 256]
        
        # Process appearance
        appearance_embedding = self.appearance_cnn(image)  # [B, 512]
        
        # Fuse modalities
        if self.fusion_method == "concat":
            fused = torch.cat([pose_embedding, appearance_embedding], dim=1)
        
        elif self.fusion_method == "add":
            pose_proj = self.pose_projection(pose_embedding)
            weight = torch.sigmoid(self.fusion_weight)
            fused = weight * pose_proj + (1 - weight) * appearance_embedding
        
        elif self.fusion_method == "attention":
            pose_proj = self.pose_projection(pose_embedding).unsqueeze(1)  # [B, 1, 256]
            appearance_proj = self.appearance_projection(appearance_embedding).unsqueeze(1)
            
            # Cross-attention: query=pose, key/value=appearance
            attended, _ = self.fusion(pose_proj, appearance_proj, appearance_proj)
            fused = attended.squeeze(1)  # [B, 256]
        
        # Classify
        logits = self.classifier(fused)
        
        return logits


class PoseOnlyClassifier(nn.Module):
    """
    Simplified model using ONLY geometric pose features.
    This is the most hand-invariant option, but loses visual details.
    Use this as a baseline to verify pose features work.
    """
    
    def __init__(self, num_classes: int, pose_feature_dim: int = 24):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(pose_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        return self.model(pose_features)


# Factory function to create appropriate model
def create_posture_model(
    num_classes: int,
    model_type: str = "hybrid",  # "hybrid", "pose_only", or "appearance_only"
    **kwargs
) -> nn.Module:
    """
    Create posture classification model.
    
    Args:
        num_classes: Number of posture classes
        model_type: Type of model to create
        **kwargs: Additional arguments for model
    
    Returns:
        PyTorch model
    """
    if model_type == "hybrid":
        return HybridPostureClassifier(num_classes, **kwargs)
    
    elif model_type == "pose_only":
        return PoseOnlyClassifier(num_classes, **kwargs)
    
    elif model_type == "appearance_only":
        # Original model (for comparison)
        from train_classifier import PostureClassifier
        return PostureClassifier(num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Test hybrid model
    batch_size = 4
    num_classes = 3
    
    # Create model
    model = HybridPostureClassifier(
        num_classes=num_classes,
        pose_feature_dim=24,
        appearance_backbone="resnet18",
        fusion_method="concat"
    )
    
    # Dummy inputs
    images = torch.randn(batch_size, 3, 224, 224)
    pose_features = torch.randn(batch_size, 24)
    
    # Forward pass
    logits = model(images, pose_features)
    
    print(f"Input image shape: {images.shape}")
    print(f"Input pose features shape: {pose_features.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
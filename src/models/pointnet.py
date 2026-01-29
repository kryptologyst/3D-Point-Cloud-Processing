"""PointNet model implementation for 3D point cloud processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TNet(nn.Module):
    """Transformation network for PointNet."""
    
    def __init__(self, k: int = 3):
        """Initialize TNet.
        
        Args:
            k: Input dimension (3 for 3D points).
        """
        super().__init__()
        self.k = k
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input points [B, N, k].
            
        Returns:
            torch.Tensor: Transformation matrix [B, k, k].
        """
        batch_size = x.size(0)
        
        # Transpose for conv1d: [B, N, k] -> [B, k, N]
        x = x.transpose(2, 1)
        
        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # FC layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        
        return x


class PointNet(nn.Module):
    """PointNet model for 3D point cloud classification and segmentation."""
    
    def __init__(
        self,
        num_classes: int = 40,
        dropout: float = 0.3,
        use_bn: bool = True,
        task: str = "classification"
    ):
        """Initialize PointNet.
        
        Args:
            num_classes: Number of output classes.
            dropout: Dropout rate.
            use_bn: Whether to use batch normalization.
            task: Task type ("classification" or "segmentation").
        """
        super().__init__()
        self.num_classes = num_classes
        self.task = task
        
        # Input transformation
        self.input_transform = TNet(k=3)
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(128) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm1d(1024) if use_bn else nn.Identity()
        
        # Feature transformation
        self.feature_transform = TNet(k=64)
        
        # Additional MLP layers
        self.conv4 = nn.Conv1d(64, 64, 1)
        self.conv5 = nn.Conv1d(64, 128, 1)
        self.conv6 = nn.Conv1d(128, 1024, 1)
        
        self.bn4 = nn.BatchNorm1d(64) if use_bn else nn.Identity()
        self.bn5 = nn.BatchNorm1d(128) if use_bn else nn.Identity()
        self.bn6 = nn.BatchNorm1d(1024) if use_bn else nn.Identity()
        
        if task == "classification":
            # Classification head
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)
            
            self.bn_fc1 = nn.BatchNorm1d(512) if use_bn else nn.Identity()
            self.bn_fc2 = nn.BatchNorm1d(256) if use_bn else nn.Identity()
            
            self.dropout = nn.Dropout(dropout)
            
        elif task == "segmentation":
            # Segmentation head
            self.fc1 = nn.Linear(1088, 512)  # 1024 + 64
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, num_classes)
            
            self.bn_fc1 = nn.BatchNorm1d(512) if use_bn else nn.Identity()
            self.bn_fc2 = nn.BatchNorm1d(256) if use_bn else nn.Identity()
            self.bn_fc3 = nn.BatchNorm1d(128) if use_bn else nn.Identity()
            
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input points [B, N, 3].
            
        Returns:
            torch.Tensor: Output logits.
        """
        batch_size, num_points, _ = x.size()
        
        # Input transformation
        trans = self.input_transform(x)
        x = torch.bmm(x, trans)
        
        # Transpose for conv1d: [B, N, 3] -> [B, 3, N]
        x = x.transpose(2, 1)
        
        # First MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Feature transformation
        trans_feat = self.feature_transform(x.transpose(2, 1))
        x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        
        # Second MLP
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        
        if self.task == "classification":
            # Global max pooling
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            
            # Classification head
            x = F.relu(self.bn_fc1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn_fc2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)
            
        elif self.task == "segmentation":
            # Concatenate global and local features
            global_feat = torch.max(x, 2, keepdim=True)[0].repeat(1, 1, num_points)
            x = torch.cat([x, global_feat], dim=1)
            
            # Transpose back: [B, C, N] -> [B, N, C]
            x = x.transpose(2, 1)
            
            # Segmentation head
            x = F.relu(self.bn_fc1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn_fc2(self.fc2(x)))
            x = self.dropout(x)
            x = F.relu(self.bn_fc3(self.fc3(x)))
            x = self.dropout(x)
            x = self.fc4(x)
        
        return x
    
    def get_transformation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute transformation regularization loss.
        
        Args:
            x: Input points [B, N, 3].
            
        Returns:
            torch.Tensor: Regularization loss.
        """
        # Input transformation
        trans = self.input_transform(x)
        
        # Feature transformation
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        trans_feat = self.feature_transform(x.transpose(2, 1))
        
        # Compute regularization loss
        I = torch.eye(3, device=x.device).view(1, 3, 3).repeat(x.size(0), 1, 1)
        loss = F.mse_loss(torch.bmm(trans, trans.transpose(2, 1)), I)
        
        I_feat = torch.eye(64, device=x.device).view(1, 64, 64).repeat(x.size(0), 1, 1)
        loss += F.mse_loss(torch.bmm(trans_feat, trans_feat.transpose(2, 1)), I_feat)
        
        return loss

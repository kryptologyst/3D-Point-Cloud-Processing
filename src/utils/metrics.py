"""3D point cloud metrics and evaluation functions."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Chamfer distance between two point clouds.
    
    Args:
        pred: Predicted point cloud [B, N, 3].
        target: Target point cloud [B, M, 3].
        
    Returns:
        torch.Tensor: Chamfer distance.
    """
    # Compute pairwise distances
    dist1 = torch.cdist(pred, target)  # [B, N, M]
    dist2 = torch.cdist(target, pred)  # [B, M, N]
    
    # Chamfer distance
    chamfer_dist = torch.mean(torch.min(dist1, dim=2)[0]) + torch.mean(torch.min(dist2, dim=2)[0])
    return chamfer_dist


def earth_mover_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Earth Mover's Distance (EMD) between two point clouds.
    
    Note: This is a simplified version. For exact EMD, use optimal transport solvers.
    
    Args:
        pred: Predicted point cloud [B, N, 3].
        target: Target point cloud [B, M, 3].
        
    Returns:
        torch.Tensor: Approximate EMD.
    """
    # Simplified EMD using Hungarian algorithm approximation
    B, N, _ = pred.shape
    M = target.shape[1]
    
    if N != M:
        # Pad or sample to make equal size
        if N > M:
            target = F.interpolate(target.transpose(1, 2), size=N, mode='linear').transpose(1, 2)
        else:
            pred = F.interpolate(pred.transpose(1, 2), size=M, mode='linear').transpose(1, 2)
            N = M
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(pred, target)  # [B, N, N]
    
    # Approximate EMD using minimum cost matching
    emd = torch.mean(torch.min(dist_matrix, dim=2)[0])
    return emd


def point_cloud_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Compute point cloud accuracy based on distance threshold.
    
    Args:
        pred: Predicted point cloud [B, N, 3].
        target: Target point cloud [B, M, 3].
        threshold: Distance threshold for accuracy.
        
    Returns:
        torch.Tensor: Accuracy score.
    """
    dist_matrix = torch.cdist(pred, target)  # [B, N, M]
    min_distances = torch.min(dist_matrix, dim=2)[0]  # [B, N]
    
    # Count points within threshold
    within_threshold = (min_distances < threshold).float()
    accuracy = torch.mean(within_threshold)
    
    return accuracy


def classification_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        predictions: Predicted logits [B, num_classes].
        targets: Ground truth labels [B].
        
    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    pred_labels = torch.argmax(predictions, dim=1).cpu().numpy()
    target_labels = targets.cpu().numpy()
    
    accuracy = accuracy_score(target_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_labels, pred_labels, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def segmentation_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Compute segmentation metrics.
    
    Args:
        predictions: Predicted logits [B, N, num_classes].
        targets: Ground truth labels [B, N].
        num_classes: Number of classes.
        
    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    pred_labels = torch.argmax(predictions, dim=2).cpu().numpy()
    target_labels = targets.cpu().numpy()
    
    # Flatten for overall metrics
    pred_flat = pred_labels.flatten()
    target_flat = target_labels.flatten()
    
    accuracy = accuracy_score(target_flat, pred_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average='weighted', zero_division=0
    )
    
    # Per-class IoU
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        target_cls = (target_flat == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'per_class_iou': ious
    }

#!/usr/bin/env python3
"""
Project 589: 3D Point Cloud Processing - Modernized Implementation

This is a modern, research-ready implementation of 3D point cloud processing
using advanced deep learning techniques. The project focuses on point cloud
classification, segmentation, and analysis using state-of-the-art models.

Key Features:
- PointNet implementation with transformation networks
- Comprehensive evaluation metrics
- Interactive Streamlit demo
- Automatic toy dataset generation
- Modern architecture with proper configuration management
- Device support: CUDA → MPS → CPU

Usage:
    python scripts/train.py                    # Train a model
    streamlit run demo/app.py                  # Run interactive demo
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf

from src.models.pointnet import PointNet
from src.data.dataset import create_dataloaders
from src.train.trainer import PointCloudTrainer
from src.utils.device import get_device, set_seed


def demonstrate_basic_processing():
    """Demonstrate basic 3D point cloud processing using Open3D."""
    print("=== Basic 3D Point Cloud Processing Demo ===")
    
    # Generate a simple point cloud for demonstration
    print("Generating synthetic point cloud...")
    points = np.random.rand(1000, 3) * 2 - 1  # Random points in [-1, 1]^3
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    print(f"Created point cloud with {len(points)} points")
    
    # Perform voxel downsampling
    print("Applying voxel downsampling...")
    voxel_size = 0.05
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    print(f"Downsampled to {len(downsampled_pcd.points)} points")
    
    # Perform plane segmentation
    print("Applying plane segmentation...")
    plane_model, inliers = downsampled_pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    
    # Extract inliers and outliers
    inlier_cloud = downsampled_pcd.select_by_index(inliers)
    outlier_cloud = downsampled_pcd.select_by_index(inliers, invert=True)
    
    print(f"Plane segmentation: {len(inliers)} inliers, {len(outlier_cloud.points)} outliers")
    
    # Color the point clouds
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red for inliers
    outlier_cloud.paint_uniform_color([0, 1.0, 0])  # Green for outliers
    
    # Save segmented point clouds
    os.makedirs("assets", exist_ok=True)
    o3d.io.write_point_cloud("assets/segmented_plane.ply", inlier_cloud)
    o3d.io.write_point_cloud("assets/segmented_objects.ply", outlier_cloud)
    print("Segmented point clouds saved to assets/")
    
    print("Basic processing demo completed!")


def demonstrate_deep_learning():
    """Demonstrate deep learning-based point cloud processing."""
    print("\n=== Deep Learning Point Cloud Processing Demo ===")
    
    # Load configuration
    config_path = "configs/config.yaml"
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        print("Config file not found, using default settings")
        config = OmegaConf.create({
            'model': {'num_classes': 4, 'dropout': 0.3, 'use_bn': True},
            'data': {'batch_size': 32, 'num_workers': 4, 'num_points': 1024, 'augment': True},
            'training': {'epochs': 10, 'learning_rate': 0.001, 'weight_decay': 1e-4, 'scheduler': 'cosine'},
            'seed': 42, 'device': 'auto'
        })
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="data",
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        num_points=config.data.num_points,
        augment=config.data.augment,
        task="classification"
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating PointNet model...")
    model = PointNet(
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
        use_bn=config.model.use_bn,
        task="classification"
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = PointCloudTrainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler_type=config.training.scheduler
    )
    
    # Train model (short training for demo)
    print("Training model...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=min(config.training.epochs, 5),  # Limit to 5 epochs for demo
        save_dir="checkpoints",
        save_every=2
    )
    
    # Test model
    print("Testing model...")
    test_metrics = trainer.validate_epoch(test_loader)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    
    print("Deep learning demo completed!")


def main():
    """Main function to run demonstrations."""
    print("3D Point Cloud Processing - Modernized Implementation")
    print("=" * 60)
    
    # Check if required packages are available
    try:
        import open3d
        import torch
        import streamlit
        print("✓ All required packages are available")
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return
    
    # Run demonstrations
    demonstrate_basic_processing()
    demonstrate_deep_learning()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("\nNext steps:")
    print("1. Run the interactive demo: streamlit run demo/app.py")
    print("2. Train a full model: python scripts/train.py")
    print("3. Evaluate a model: python scripts/evaluate.py --checkpoint checkpoints/best_model.pth")
    print("4. View the README.md for detailed documentation")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Main training script for 3D point cloud processing."""

import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data.dataset import create_dataloaders
from src.models.pointnet import PointNet
from src.train.trainer import PointCloudTrainer
from src.utils.device import get_device, set_seed


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train 3D point cloud model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Set seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
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
    print("Creating model...")
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
        scheduler_type=config.training.scheduler,
        warmup_epochs=config.training.warmup_epochs
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        save_dir=args.checkpoint_dir,
        save_every=10
    )
    
    # Test model
    print("Testing model...")
    test_metrics = trainer.validate_epoch(test_loader)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()

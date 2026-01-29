#!/usr/bin/env python3
"""Evaluation script for 3D point cloud models."""

import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data.dataset import create_dataloaders
from src.models.pointnet import PointNet
from src.utils.device import get_device, set_seed
from src.utils.metrics import classification_metrics, chamfer_distance


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate 3D point cloud model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Path to output directory"
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
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        num_points=config.data.num_points,
        augment=False,  # No augmentation for evaluation
        task="classification"
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = PointNet(
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
        use_bn=config.model.use_bn,
        task="classification"
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluate model
    print("Evaluating model...")
    all_predictions = []
    all_targets = []
    all_points = []
    
    with torch.no_grad():
        for batch in test_loader:
            points = batch['points'].to(device)
            targets = batch['class_id'].squeeze().to(device)
            
            outputs = model(points)
            
            all_predictions.append(outputs)
            all_targets.append(targets)
            all_points.append(points)
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_points = torch.cat(all_points, dim=0)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = classification_metrics(all_predictions, all_targets)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1-Score: {metrics['f1']:.4f}")
    print("="*50)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    import json
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    predictions_data = {
        'predictions': all_predictions.cpu().numpy().tolist(),
        'targets': all_targets.cpu().numpy().tolist(),
        'points': all_points.cpu().numpy().tolist()
    }
    
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(predictions_data, f, indent=2)
    
    print(f"Results saved to {args.output_dir}")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()

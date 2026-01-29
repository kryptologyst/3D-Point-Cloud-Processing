"""Training module for 3D point cloud models."""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.pointnet import PointNet
from src.utils.device import get_device, set_seed
from src.utils.metrics import classification_metrics


class PointCloudTrainer:
    """Trainer for 3D point cloud models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 10
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            device: Device to use for training.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            scheduler_type: Type of learning rate scheduler.
            warmup_epochs: Number of warmup epochs.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = self._get_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = []
        self.val_history = []
        
    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler."""
        if self.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=200, eta_min=1e-6
            )
        elif self.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.1
            )
        elif self.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=10
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Dict[str, float]: Training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            points = batch['points'].to(self.device)
            targets = batch['class_id'].squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(points)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Add transformation loss if available
            if hasattr(self.model, 'get_transformation_loss'):
                trans_loss = self.model.get_transformation_loss(points)
                loss += 0.001 * trans_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_targets.append(targets.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = classification_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Dict[str, float]: Validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                # Move to device
                points = batch['points'].to(self.device)
                targets = batch['class_id'].squeeze().to(self.device)
                
                # Forward pass
                outputs = self.model(points)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                all_predictions.append(outputs)
                all_targets.append(targets)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = classification_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        save_dir: str = "checkpoints",
        save_every: int = 10
    ) -> None:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train.
            save_dir: Directory to save checkpoints.
            save_every: Save checkpoint every N epochs.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            self.val_history.append(val_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(
                    os.path.join(save_dir, "best_model.pth"),
                    epoch, val_metrics
                )
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                    epoch, val_metrics
                )
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    
    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint.
            epoch: Current epoch.
            metrics: Current metrics.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Metrics: {checkpoint['metrics']}")

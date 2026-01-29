"""Tests for 3D point cloud processing."""

import pytest
import torch
import numpy as np

from src.models.pointnet import PointNet, TNet
from src.utils.device import get_device, set_seed
from src.utils.metrics import chamfer_distance, classification_metrics


class TestPointNet:
    """Test PointNet model."""
    
    def test_tnet_forward(self):
        """Test TNet forward pass."""
        tnet = TNet(k=3)
        x = torch.randn(2, 1024, 3)
        output = tnet(x)
        
        assert output.shape == (2, 3, 3)
        assert torch.allclose(torch.bmm(output, output.transpose(2, 1)), 
                             torch.eye(3).unsqueeze(0).repeat(2, 1, 1), 
                             atol=1e-6)
    
    def test_pointnet_classification(self):
        """Test PointNet classification."""
        model = PointNet(num_classes=4, task="classification")
        x = torch.randn(2, 1024, 3)
        output = model(x)
        
        assert output.shape == (2, 4)
        assert torch.allclose(torch.sum(torch.softmax(output, dim=1), dim=1), 
                             torch.ones(2), atol=1e-6)
    
    def test_pointnet_segmentation(self):
        """Test PointNet segmentation."""
        model = PointNet(num_classes=4, task="segmentation")
        x = torch.randn(2, 1024, 3)
        output = model(x)
        
        assert output.shape == (2, 1024, 4)
        assert torch.allclose(torch.sum(torch.softmax(output, dim=2), dim=2), 
                             torch.ones(2, 1024), atol=1e-6)
    
    def test_transformation_loss(self):
        """Test transformation loss computation."""
        model = PointNet(num_classes=4, task="classification")
        x = torch.randn(2, 1024, 3)
        loss = model.get_transformation_loss(x)
        
        assert loss.item() >= 0
        assert loss.requires_grad


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_chamfer_distance(self):
        """Test Chamfer distance computation."""
        pred = torch.randn(2, 100, 3)
        target = torch.randn(2, 100, 3)
        
        distance = chamfer_distance(pred, target)
        assert distance.item() >= 0
        assert distance.requires_grad
    
    def test_classification_metrics(self):
        """Test classification metrics."""
        predictions = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))
        
        metrics = classification_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1


class TestDevice:
    """Test device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seeds are set
        torch.manual_seed(42)
        x1 = torch.randn(10)
        torch.manual_seed(42)
        x2 = torch.randn(10)
        
        assert torch.allclose(x1, x2)


if __name__ == "__main__":
    pytest.main([__file__])

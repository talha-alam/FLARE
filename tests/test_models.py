import pytest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required packages can be imported."""
    import torch
    import torchvision
    import PIL
    import numpy as np
    assert True

def test_device_availability():
    """Test CUDA availability (will pass even if CUDA is not available)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert isinstance(device, torch.device)

@pytest.mark.skip(reason="Model weights not available in CI")
def test_model_loading():
    """Test model loading (skipped in CI)."""
    from models import build_classifier
    model = build_classifier(
        model_name="resnet50",
        num_classes=8,
        pretrained=False
    )
    assert isinstance(model, torch.nn.Module)
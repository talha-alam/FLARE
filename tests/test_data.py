import pytest
import torch
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_transforms():
    """Test data transformations."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    
    # Apply transformation
    tensor_image = transform(pil_image)
    
    assert isinstance(tensor_image, torch.Tensor)
    assert tensor_image.shape == (3, 224, 224)

@pytest.mark.skip(reason="Dataset not available in CI")
def test_dataset():
    """Test dataset loading (skipped in CI)."""
    from data.dataset import SpaceNetDataset
    
    dataset = SpaceNetDataset(
        root_dir="path/to/data",
        transform=None,
        phase='train'
    )
    assert len(dataset) >= 0
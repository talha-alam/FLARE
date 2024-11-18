import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from typing import Optional, Tuple, List

class SpaceNetDataset(Dataset):
    """SpaceNet Dataset for astronomical image classification."""
    
    def __init__(
        self, 
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        phase: str = 'train',
        fine_grained: bool = True
    ):
        """
        Args:
            root_dir (str): Root directory of the dataset
            transform (callable, optional): Transform to be applied on images
            phase (str): 'train', 'val', or 'test'
            fine_grained (bool): If True, use 8 classes, else use 4 macro classes
        """
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        self.fine_grained = fine_grained
        
        # Define class mappings
        self.fine_grained_classes = [
            'planets', 'galaxies', 'asteroids', 'nebulae',
            'comets', 'black_holes', 'stars', 'constellations'
        ]
        
        self.macro_classes = {
            'Astronomical_Patterns': ['nebulae', 'constellations'],
            'Celestial_Bodies': ['planets', 'asteroids'],
            'Cosmic_Phenomena': ['black_holes', 'comets'],
            'Stellar_Objects': ['stars', 'galaxies']
        }
        
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(
                self.fine_grained_classes if fine_grained 
                else list(self.macro_classes.keys())
            )
        }
        
        self.samples = self._build_dataset()
        
    def _build_dataset(self) -> List[Tuple[str, int]]:
        """Build dataset by collecting all image paths and labels."""
        samples = []
        
        if self.fine_grained:
            for class_name in self.fine_grained_classes:
                class_dir = os.path.join(self.root_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                    
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        samples.append((img_path, self.class_to_idx[class_name]))
        else:
            for macro_class, fine_classes in self.macro_classes.items():
                for class_name in fine_classes:
                    class_dir = os.path.join(self.root_dir, class_name)
                    if not os.path.exists(class_dir):
                        continue
                        
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(class_dir, img_name)
                            samples.append((img_path, self.class_to_idx[macro_class]))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def build_dataloaders(
    config,
    root_dir: str,
    fine_grained: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test dataloaders.
    
    Args:
        config: Configuration object containing data settings
        root_dir: Root directory of the dataset
        fine_grained: Whether to use fine-grained or macro classification
        
    Returns:
        tuple: Train, validation, and test dataloaders
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.6,
            contrast=0.4,
            saturation=0.3,
            hue=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    train_dataset = SpaceNetDataset(
        root_dir=root_dir,
        transform=train_transform,
        phase='train',
        fine_grained=fine_grained
    )
    
    val_dataset = SpaceNetDataset(
        root_dir=root_dir,
        transform=test_transform,
        phase='val',
        fine_grained=fine_grained
    )
    
    test_dataset = SpaceNetDataset(
        root_dir=root_dir,
        transform=test_transform,
        phase='test',
        fine_grained=fine_grained
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
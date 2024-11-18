import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List

class AugmentationPipeline:
    """Data augmentation pipeline for astronomical images."""
    
    def __init__(self, config):
        self.config = config
        
    def get_train_transforms(self) -> transforms.Compose:
        """Get training data transforms."""
        return transforms.Compose([
            transforms.Resize((self.config.data.image_size, self.config.data.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.6,
                contrast=0.4,
                saturation=0.3,
                hue=0.2
            ),
            transforms.RandomResizedCrop(
                self.config.data.image_size,
                scale=(0.75, 0.90)
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_eval_transforms(self) -> transforms.Compose:
        """Get evaluation data transforms."""
        return transforms.Compose([
            transforms.Resize((self.config.data.image_size, self.config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
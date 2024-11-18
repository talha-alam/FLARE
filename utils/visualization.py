# utils/visualization.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import io
from typing import Optional, List, Tuple

def visualize_samples(original_images: torch.Tensor,
                     hr_images: torch.Tensor,
                     synthetic_images: torch.Tensor,
                     nrow: int = 4) -> Image.Image:
    """Visualize original, HR, and synthetic images in a grid."""
    # Denormalize images
    def denorm(x):
        return x.mul(0.5).add(0.5).clamp(0, 1)
    
    # Create grid for each type
    orig_grid = make_grid(denorm(original_images), nrow=nrow)
    hr_grid = make_grid(denorm(hr_images), nrow=nrow)
    syn_grid = make_grid(denorm(synthetic_images), nrow=nrow)
    
    # Combine grids vertically
    final_grid = torch.cat([orig_grid, hr_grid, syn_grid], dim=1)
    
    # Convert to PIL Image
    img = Image.fromarray(
        (final_grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )
    
    return img

def plot_confusion_matrix(confusion_mat: np.ndarray,
                         class_names: List[str],
                         title: str = 'Confusion Matrix') -> Image.Image:
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    
    return img

def plot_training_curves(metrics: dict,
                        save_path: Optional[str] = None) -> None:
    """Plot training curves."""
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 3, 2)
    plt.plot(metrics['train_accuracy'], label='Train Acc')
    plt.plot(metrics['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 scores
    plt.subplot(1, 3, 3)
    plt.plot(metrics['train_f1'], label='Train F1')
    plt.plot(metrics['val_f1'], label='Val F1')
    plt.title('F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
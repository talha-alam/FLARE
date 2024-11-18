# utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Union
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(predictions: Union[List, np.ndarray], 
                     targets: Union[List, np.ndarray]) -> Dict[str, float]:
    """Calculate classification metrics."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average='weighted'),
        'recall': recall_score(targets, predictions, average='weighted'),
        'f1': f1_score(targets, predictions, average='weighted'),
    }
    
    return metrics

def calculate_image_metrics(original: np.ndarray, 
                          generated: np.ndarray) -> Dict[str, float]:
    """Calculate image quality metrics."""
    psnr_value = psnr(original, generated)
    ssim_value = ssim(original, generated, multichannel=True)
    
    return {
        'psnr': psnr_value,
        'ssim': ssim_value
    }

def calculate_class_distribution(labels: Union[List, np.ndarray]) -> Dict[str, int]:
    """Calculate class distribution in dataset."""
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional

def preprocess_image(image: Image.Image,
                    target_size: Tuple[int, int] = (512, 512),
                    normalize: bool = True) -> np.ndarray:
    """Preprocess image for model input."""
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize if required
    if normalize:
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    return image

def center_crop(image: np.ndarray,
                target_size: Tuple[int, int]) -> np.ndarray:
    """Center crop the image to target size."""
    h, w = image.shape[:2]
    th, tw = target_size
    
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    
    return image[i:i+th, j:j+tw]

def adjust_dynamic_range(image: np.ndarray,
                        percentile_low: float = 1,
                        percentile_high: float = 99) -> np.ndarray:
    """Adjust dynamic range of astronomical images."""
    low = np.percentile(image, percentile_low)
    high = np.percentile(image, percentile_high)
    
    image_adjusted = np.clip((image - low) / (high - low), 0, 1)
    return image_adjusted
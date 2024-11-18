import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

def build_classifier(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """Build a classifier model."""
    
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "vit":
        model = models.vit_b_16(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    
    return model
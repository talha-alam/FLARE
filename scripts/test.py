import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from data import build_dataloaders
from models import build_classifier
from utils import calculate_metrics
from configs import get_config

def test(args):
    # Load config
    config = get_config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    model = build_classifier(
        model_name=config.model.backbone,
        num_classes=config.model.num_classes,
        checkpoint_path=args.checkpoint
    ).to(device)
    
    # Get dataloaders
    _, _, test_loader = build_dataloaders(
        config,
        root_dir=args.data,
        fine_grained=args.fine_grained
    )
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_labels)
    
    # Print results
    print("\nTest Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to test data")
    parser.add_argument("--fine_grained", action="store_true",
                        help="Use fine-grained classification")
    args = parser.parse_args()
    
    test(args)
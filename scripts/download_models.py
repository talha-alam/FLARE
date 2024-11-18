# scripts/download_models.py

import os
import requests
from tqdm import tqdm
import hashlib
from typing import Dict, Optional

MODEL_URLS = {
    'swinir': 'https://github.com/talha-alam/FLARE/releases/download/v1.0/swinir_spacenet.pth',
    'unidiffuser': 'https://github.com/talha-alam/FLARE/releases/download/v1.0/unidiffuser_spacenet.pth',
    'resnet50_fine': 'https://github.com/talha-alam/FLARE/releases/download/v1.0/resnet50_fine.pth',
    'resnet50_macro': 'https://github.com/talha-alam/FLARE/releases/download/v1.0/resnet50_macro.pth'
}

MODEL_CHECKSUMS = {
    'swinir': 'checksum_here',
    'unidiffuser': 'checksum_here',
    'resnet50_fine': 'checksum_here',
    'resnet50_macro': 'checksum_here'
}

def download_file(url: str, filename: str) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def verify_checksum(filename: str, expected_checksum: str) -> bool:
    """Verify file checksum."""
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_checksum

def download_models(model_dir: str = 'pretrained_models',
                   models: Optional[list] = None) -> None:
    """Download pretrained models."""
    os.makedirs(model_dir, exist_ok=True)
    
    if models is None:
        models = list(MODEL_URLS.keys())
    
    for model in models:
        if model not in MODEL_URLS:
            print(f"Unknown model: {model}")
            continue
            
        filename = os.path.join(model_dir, f"{model}.pth")
        
        # Skip if file exists and checksum matches
        if os.path.exists(filename) and verify_checksum(filename, MODEL_CHECKSUMS[model]):
            print(f"{model} already downloaded and verified.")
            continue
            
        print(f"Downloading {model}...")
        download_file(MODEL_URLS[model], filename)
        
        # Verify checksum
        if verify_checksum(filename, MODEL_CHECKSUMS[model]):
            print(f"{model} successfully downloaded and verified.")
        else:
            print(f"Warning: Checksum mismatch for {model}!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='pretrained_models',
                      help='Directory to save pretrained models')
    parser.add_argument('--models', nargs='+', help='Specific models to download')
    args = parser.parse_args()
    
    download_models(args.model_dir, args.models)
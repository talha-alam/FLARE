import torch
import argparse
from pathlib import Path
from models import SwinIR, DiffusionModel
from utils import visualize_samples
from PIL import Image
import torchvision.transforms as transforms

def generate_samples(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    swinir = SwinIR.from_pretrained(args.swinir_path).to(device)
    diffusion = DiffusionModel.from_pretrained(args.diffusion_path).to(device)
    
    # Load and preprocess input image
    image = Image.open(args.input_image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    # Generate samples
    with torch.no_grad():
        # Generate HR image
        hr_image = swinir(image)
        
        # Generate synthetic samples
        synthetic_images = []
        for _ in range(args.num_samples):
            synthetic = diffusion.sample(
                batch_size=1,
                prompt=args.prompt,
                device=device
            )
            synthetic_images.append(synthetic)
        
        synthetic_images = torch.cat(synthetic_images, dim=0)
    
    # Visualize and save results
    visualization = visualize_samples(
        original_images=image,
        hr_images=hr_image,
        synthetic_images=synthetic_images
    )
    
    # Save results
    output_path = Path(args.output_dir) / f"samples_{Path(args.input_image).stem}.png"
    visualization.save(output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--swinir_path", type=str, required=True,
                        help="Path to SwinIR checkpoint")
    parser.add_argument("--diffusion_path", type=str, required=True,
                        help="Path to Diffusion model checkpoint")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for generation")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of samples to generate")
    args = parser.parse_args()
    
    generate_samples(args)
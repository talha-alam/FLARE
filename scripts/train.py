import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from typing import Optional, Dict, Any

from models.restoration import SwinIR
from models.diffusion import DiffusionModel
from data.dataset import build_dataloaders
from utils.metrics import calculate_metrics
from utils.visualization import visualize_samples

class FLARETrainer:
    """Trainer for FLARE framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.swinir = SwinIR(
            img_size=config.swinir.img_size,
            patch_size=config.swinir.patch_size,
            embed_dim=config.swinir.embed_dim,
            depths=config.swinir.depths,
            num_heads=config.swinir.num_heads
        ).to(self.device)
        
        self.diffusion = DiffusionModel(
            img_size=config.diffusion.img_size,
            model_channels=config.diffusion.model_channels,
            num_heads=config.diffusion.num_heads
        ).to(self.device)
        
        # Initialize classifier
        self.classifier = self._build_classifier().to(self.device)
        
        # Setup optimizers
        self.swinir_optimizer = optim.Adam(
            self.swinir.parameters(),
            lr=config.training.learning_rate
        )
        
        self.diffusion_optimizer = optim.Adam(
            self.diffusion.parameters(),
            lr=config.training.learning_rate
        )
        
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=config.training.learning_rate
        )
        
        # Initialize dataloaders
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            config,
            root_dir=config.data.root_dir,
            fine_grained=config.data.fine_grained
        )
        
        # Setup logging
        if config.logging.wandb:
            wandb.init(project=config.logging.project_name)
            wandb.config.update(config)
            
    def _build_classifier(self) -> nn.Module:
        """Build the classifier model."""
        if self.config.model.backbone == "resnet50":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.config.model.num_classes)
        else:
            raise NotImplementedError(f"Unknown backbone: {self.config.model.backbone}")
        return model
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.swinir.train()
        self.diffusion.train()
        self.classifier.train()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(self.train_loader)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)
            
            # Step 1: Generate HR images
            hr_images = self.swinir(images)
            
            # Step 2: Generate synthetic samples using diffusion
            noise = torch.randn_like(hr_images)
            t = torch.randint(0, len(self.diffusion.betas), (batch_size,), device=self.device)
            noisy_images = self.diffusion.q_sample(hr_images, t, noise)
            predicted_noise = self.diffusion(noisy_images, t)
            diffusion_loss = nn.functional.mse_loss(predicted_noise, noise)
            
            # Step 3: Classification
            combined_images = torch.cat([hr_images, predicted_noise], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)
            
            logits = self.classifier(combined_images)
            classification_loss = nn.CrossEntropyLoss()(logits, combined_labels)
            
            # Total loss
            total_loss_batch = diffusion_loss + classification_loss
            
            # Optimization step
            self.swinir_optimizer.zero_grad()
            self.diffusion_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            
            total_loss_batch.backward()
            
            self.swinir_optimizer.step()
            self.diffusion_optimizer.step()
            self.classifier_optimizer.step()
            
            # Update statistics
            total_loss += total_loss_batch.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_description(
                f"Train Loss: {total_loss_batch.item():.4f} "
                f"Diff Loss: {diffusion_loss.item():.4f} "
                f"Class Loss: {classification_loss.item():.4f}"
            )
            
        epoch_loss = total_loss / total_samples
        
        metrics = {
            'train_loss': epoch_loss,
            'train_diff_loss': diffusion_loss.item(),
            'train_class_loss': classification_loss.item()
        }
        
        return metrics
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.swinir.eval()
        self.diffusion.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)
            
            # Generate HR images
            hr_images = self.swinir(images)
            
            # Generate synthetic samples
            synthetic_images = self.diffusion.sample(
                batch_size=batch_size,
                device=self.device
            )
            
            # Combine real and synthetic images
            combined_images = torch.cat([hr_images, synthetic_images], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)
            
            # Classification
            logits = self.classifier(combined_images)
            loss = nn.CrossEntropyLoss()(logits, combined_labels)
            
            # Update statistics
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Store predictions for metric calculation
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(combined_labels.cpu().numpy())
            
        # Calculate metrics
        metrics = calculate_metrics(all_preds, all_labels)
        metrics['val_loss'] = total_loss / total_samples
        
        return metrics
        
    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Test the model."""
        self.swinir.eval()
        self.diffusion.eval()
        self.classifier.eval()
        
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(self.test_loader, desc="Testing"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Generate HR images
            hr_images = self.swinir(images)
            
            # Classification
            logits = self.classifier(hr_images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        # Calculate metrics
        metrics = calculate_metrics(all_preds, all_labels)
        
        return metrics
        
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'swinir_state_dict': self.swinir.state_dict(),
            'diffusion_state_dict': self.diffusion.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'swinir_optimizer': self.swinir_optimizer.state_dict(),
            'diffusion_optimizer': self.diffusion_optimizer.state_dict(),
            'classifier_optimizer': self.classifier_optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.logging.log_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.logging.log_dir,
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.swinir.load_state_dict(checkpoint['swinir_state_dict'])
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        self.swinir_optimizer.load_state_dict(checkpoint['swinir_optimizer'])
        self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer'])
        self.classifier_optimizer.load_state_dict(checkpoint['classifier_optimizer'])
        
        return checkpoint['epoch'], checkpoint['metrics']
        
    def train(self):
        """Main training loop."""
        best_val_metric = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.training.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Testing
            test_metrics = self.test()
            
            # Log metrics
            metrics = {
                **train_metrics,
                **val_metrics,
                **test_metrics,
                'epoch': epoch
            }
            
            if self.config.logging.wandb:
                wandb.log(metrics)
                
            # Save checkpoint and check for early stopping
            is_best = val_metrics['val_loss'] < best_val_metric
            if is_best:
                best_val_metric = val_metrics['val_loss']
                patience_counter = 0
            else:
                patience_counter += 1
                
            self.save_checkpoint(epoch, metrics, is_best)
            
            # Generate and log sample images
            if (epoch + 1) % self.config.logging.save_frequency == 0:
                self.generate_samples(epoch)
                
            # Early stopping
            if patience_counter >= self.config.training.patience:
                print("Early stopping triggered!")
                break
                
    def generate_samples(self, epoch: int):
        """Generate and save sample images."""
        self.swinir.eval()
        self.diffusion.eval()
        
        with torch.no_grad():
            # Get a batch of images
            images, _ = next(iter(self.val_loader))
            images = images.to(self.device)
            
            # Generate HR images
            hr_images = self.swinir(images)
            
            # Generate synthetic samples
            synthetic_images = self.diffusion.sample(
                batch_size=images.size(0),
                device=self.device
            )
            
            # Visualize and save
            visualization = visualize_samples(
                original_images=images,
                hr_images=hr_images,
                synthetic_images=synthetic_images
            )
            
            save_path = os.path.join(
                self.config.logging.log_dir,
                f'samples_epoch_{epoch}.png'
            )
            visualization.save(save_path)
            
            if self.config.logging.wandb:
                wandb.log({
                    'samples': wandb.Image(visualization),
                    'epoch': epoch
                })

if __name__ == "__main__":
    import argparse
    from configs.default_config import get_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.py')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Create trainer
    trainer = FLARETrainer(config)
    
    # Resume from checkpoint if specified
    if args.checkpoint:
        start_epoch, metrics = trainer.load_checkpoint(args.checkpoint)
        print(f"Resuming from epoch {start_epoch}")
    
    # Start training
    trainer.train()
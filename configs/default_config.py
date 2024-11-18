from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    # Data settings
    config.data = ConfigDict()
    config.data.image_size = 512
    config.data.num_workers = 4
    config.data.batch_size = 32
    config.data.train_ratio = 0.7
    config.data.valid_ratio = 0.15
    
    # Model settings
    config.model = ConfigDict()
    config.model.num_classes = 8  # For fine-grained classification
    config.model.backbone = "resnet50"
    config.model.pretrained = True
    
    # SwinIR settings
    config.swinir = ConfigDict()
    config.swinir.img_size = 64
    config.swinir.patch_size = 2
    config.swinir.embed_dim = 1536
    config.swinir.depth = 30
    config.swinir.num_heads = 24
    
    # UniDiffuser settings
    config.diffusion = ConfigDict()
    config.diffusion.steps = 50
    config.diffusion.cfg_scale = 8.0
    config.diffusion.n_samples = 4
    
    # Augmentation settings
    config.augmentation = ConfigDict()
    config.augmentation.alpha = 0.5  # Weight for traditional augmentations
    config.augmentation.beta = 1.0   # Weight for diffusion samples
    
    # Training settings
    config.training = ConfigDict()
    config.training.num_epochs = 50
    config.training.learning_rate = 2e-5
    config.training.weight_decay = 0.01
    config.training.optimizer = "adam"
    config.training.scheduler = "cosine"
    config.training.warmup_epochs = 5
    config.training.patience = 10
    
    # Logging settings
    config.logging = ConfigDict()
    config.logging.save_frequency = 5
    config.logging.log_dir = "experiments/logs"
    config.logging.wandb = True
    config.logging.project_name = "FLARE"
    
    return config
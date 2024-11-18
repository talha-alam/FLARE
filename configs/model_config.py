from ml_collections import ConfigDict

def get_swinir_config() -> ConfigDict:
    """SwinIR model configuration."""
    config = ConfigDict()
    
    # Model architecture
    config.img_size = 64
    config.patch_size = 1
    config.in_chans = 3
    config.embed_dim = 96
    config.depths = [6, 6, 6, 6]
    config.num_heads = [6, 6, 6, 6]
    config.window_size = 7
    config.mlp_ratio = 4.
    config.qkv_bias = True
    config.drop_rate = 0.
    config.attn_drop_rate = 0.
    config.drop_path_rate = 0.1
    config.patch_norm = True
    config.use_checkpoint = False
    config.upscale = 2
    config.img_range = 1.
    config.upsampler = 'pixelshuffle'
    config.resi_connection = '1conv'
    
    return config

def get_diffusion_config() -> ConfigDict:
    """Diffusion model configuration."""
    config = ConfigDict()
    
    # Model architecture
    config.img_size = 64
    config.in_channels = 4
    config.model_channels = 1536
    config.num_heads = 24
    config.text_dim = 64
    config.num_text_tokens = 77
    config.clip_img_dim = 512
    config.use_checkpoint = True
    
    # Diffusion process
    config.beta_schedule = 'linear'
    config.beta_start = 0.00085
    config.beta_end = 0.0120
    config.num_timesteps = 1000
    
    # Training
    config.batch_size = 16
    config.learning_rate = 2e-5
    config.weight_decay = 0.01
    config.num_epochs = 50
    config.warmup_epochs = 5
    
    return config

def get_classifier_config() -> ConfigDict:
    """Classifier model configuration."""
    config = ConfigDict()
    
    # Model architecture
    config.backbone = 'resnet50'  # ['resnet50', 'densenet121', 'vit']
    config.pretrained = True
    config.num_classes = 8  # 8 for fine-grained, 4 for macro
    config.dropout_rate = 0.1
    
    # Training
    config.batch_size = 32
    config.learning_rate = 1e-4
    config.weight_decay = 0.01
    config.num_epochs = 100
    config.warmup_epochs = 5
    
    # Data augmentation
    config.aug_strength = 0.5
    config.mixup_alpha = 0.2
    config.cutmix_alpha = 1.0
    
    return config

def get_full_config() -> ConfigDict:
    """Get full model configuration."""
    config = ConfigDict()
    
    config.swinir = get_swinir_config()
    config.diffusion = get_diffusion_config()
    config.classifier = get_classifier_config()
    
    return config
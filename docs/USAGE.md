# Usage Guide

## Dataset

### Downloading SpaceNet
The SpaceNet dataset is available on [Kaggle](https://www.kaggle.com/datasets/razaimam45/spacenet-an-optimally-distributed-astronomy-data/).

```bash
# Using Kaggle API
kaggle datasets download razaimam45/spacenet-an-optimally-distributed-astronomy-data

# Extract dataset
unzip spacenet-an-optimally-distributed-astronomy-data.zip -d data/
```

### Data Structure
```
data/
├── fine_grained/
│   ├── planets/
│   ├── galaxies/
│   ├── asteroids/
│   ├── nebulae/
│   ├── comets/
│   ├── black_holes/
│   ├── stars/
│   └── constellations/
└── macro/
    ├── Astronomical_Patterns/
    ├── Celestial_Bodies/
    ├── Cosmic_Phenomena/
    └── Stellar_Objects/
```

## Training

### Configuration
1. Modify configs in `configs/default_config.py`
2. Choose model architecture in `configs/model_config.py`

### Training Commands
```bash
# Train from scratch
python scripts/train.py --config configs/default_config.py

# Resume training
python scripts/train.py --config configs/default_config.py --checkpoint path/to/checkpoint.pth

# Fine-grained classification
python scripts/train.py --config configs/default_config.py --fine_grained

# Mixed precision training
python scripts/train.py --config configs/default_config.py --amp
```

### Monitoring Training
```bash
# Start tensorboard
tensorboard --logdir experiments/logs/

# Monitor on W&B
wandb login
python scripts/train.py --wandb
```

## Evaluation

### Testing Models
```bash
python scripts/test.py --checkpoint pretrained_models/resnet50_fine.pth --data path/to/test/data
```

### Generate Samples
```bash
python scripts/generate_samples.py \
    --input_image path/to/image.jpg \
    --prompt "your text prompt" \
    --num_samples 4
```

## Using Pretrained Models

### Image Restoration
```python
from models import SwinIR

# Load model
model = SwinIR.from_pretrained('pretrained_models/swinir_spacenet.pth')

# Process image
hr_image = model.enhance(input_image)
```

### Image Generation
```python
from models import DiffusionModel

# Load model
model = DiffusionModel.from_pretrained('pretrained_models/unidiffuser_spacenet.pth')

# Generate samples
samples = model.sample(prompt="your text prompt", num_samples=4)
```

### Classification
```python
from models import build_classifier

# Load model
model = build_classifier(
    model_name="resnet50",
    num_classes=8,
    checkpoint_path="pretrained_models/resnet50_fine.pth"
)

# Classify image
prediction = model.predict(image)
```

## Best Practices

1. Data Preprocessing
   - Center crop images to 512x512
   - Normalize using ImageNet statistics
   - Apply augmentations during training

2. Training Tips
   - Start with small learning rate (2e-5)
   - Use learning rate warmup
   - Monitor validation loss for early stopping
   - Save checkpoints regularly

3. Inference
   - Use model.eval() mode
   - Apply test-time augmentation for better results
   - Ensemble multiple models for best performance
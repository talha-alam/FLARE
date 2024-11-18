# Installation Guide

## Requirements
- Python 3.8 or higher
- CUDA 11.0 or higher (for GPU support)
- 16GB RAM minimum (32GB recommended)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/FLARE.git
cd FLARE
```

### 2. Create a Conda Environment
```bash
conda create -n flare python=3.8
conda activate flare
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pretrained Models
```bash
python scripts/download_models.py
```

### 5. Install as Package (Optional)
```bash
pip install -e .
```

## Common Issues

### CUDA Out of Memory
If you encounter CUDA out of memory errors:
1. Reduce batch size in config
2. Use gradient checkpointing
3. Use mixed precision training

### Package Dependencies
If you encounter dependency conflicts:
1. Install PyTorch separately first:
```bash
conda install pytorch torchvision -c pytorch
```
2. Then install other dependencies:
```bash
pip install -r requirements.txt
```

## Verification
To verify the installation:
```bash
python tests/test_models.py
```

## Additional Notes
- For training on multiple GPUs, install apex for mixed precision training
- For logging experiments, create a Weights & Biases account
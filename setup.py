from setuptools import setup, find_packages

setup(
    name="flare",
    version="1.0.0",
    description="FLARE: Diffusion-based Augmentation Method in Astronomical Imaging",
    author="Mohammed Talha Alam, Raza Imam",
    author_email="mohammed.alam@mbzuai.ac.ae",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "timm>=0.4.12",
        "ml_collections>=0.1.1",
        "numpy>=1.19.2",
        "Pillow>=8.3.1",
        "scikit-image>=0.18.3",
        "scikit-learn>=0.24.2",
        "tqdm>=4.62.3",
        "wandb>=0.12.0"
    ],
    python_requires=">=3.8",
)
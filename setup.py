from setuptools import setup, find_packages

setup(
    name="decision-transformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "wandb",
        "hydra-core",
        "transformers",
        "safetensors",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "black",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-dt=scripts.training:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A Decision Transformer implementation for game input prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/decision-transformer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

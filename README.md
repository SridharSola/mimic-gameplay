# Decision Transformer for Game Input Prediction

This project implements a Decision Transformer model for predicting game inputs based on state observations. It's designed to learn from gameplay data and generate mouse and keyboard actions.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/decision-transformer.git
   cd decision-transformer
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n dt-env python=3.8
   conda activate dt-env
   ```

3. Install the required packages:
   ```bash
   pip install -e .
   ```

## Data Preparation

Convert your CSV gameplay data to SafeTensor format using the provided script:

```bash
python scripts/preprocess_data.py --input_file path/to/your/csv/file.csv --output_file path/to/output/safetensors/file.safetensors
```


## Configuration

Adjust the hyperparameters and settings in `configs/config.yaml`:


## Training

To train the model, run:

```bash
python scripts/training.py
```

## Model Architecture

The Decision Transformer is based on GPT-2 and processes state observations, mouse actions, and key actions to predict future actions. Key components:

1. `DecisionTransformer`: Main model class
2. `GPT2Model`: Underlying transformer architecture
3. `Trainer`: Handles the training loop and evaluation

## Customization

- Modify `decision_transformer/models/decision_transformer.py` to adjust the model architecture.
- Update `decision_transformer/training/trainer.py` to change the training process.
- Alter `data/data_loader.py` to modify how data is loaded and processed.


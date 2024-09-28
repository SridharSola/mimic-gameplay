import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from decision_transformer.models.decision_transformer import DecisionTransformer
from data.dataset import get_dataset
from decision_transformer.training.dt_trainer import DTTrainer
import wandb
import hydra
import os
from omegaconf import DictConfig
from pathlib import Path

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Get the absolute path of the script
    script_dir = Path(__file__).parent.absolute()
    
    # Construct the absolute path to the data directory
    data_dir = script_dir.parent / cfg.data_dir
    
    print(f"Absolute data directory path: {data_dir}")
    print(f"Data directory exists: {os.path.exists(data_dir)}")

    wandb.init()

    # Get dataset
    dataset = get_dataset(str(data_dir), cfg.sequence_length, cfg.chunk_size)
    print("dataset 1: ", dataset[1000])
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Get data stats
    data_stats = dataset.get_data_stats()

    # Model initialization
    state_dim = dataset[0]['input_states'].shape[-1]
    mouse_dim = dataset[0]['input_mouse_actions'].shape[-1]
    key_dim = dataset.get_key_dim()

    model = DecisionTransformer(
        state_dim=state_dim,
        mouse_dim=mouse_dim,
        key_dim=key_dim,
        hidden_size=cfg.hidden_size,
        max_length=cfg.sequence_length,
        max_ep_len=cfg.max_ep_len,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        chunk_size=cfg.chunk_size
    )

    # print model summary
    print(model)

    # print model parameters of last layer
    print(model.transformer.h[-1].attn.c_attn.weight)
    # Loss function
    def loss_fn(mouse_preds, key_preds, mouse_targets, key_targets):
        mouse_loss = nn.MSELoss()(mouse_preds, mouse_targets)
        key_loss = nn.BCEWithLogitsLoss()(key_preds, key_targets)
        total_loss = mouse_loss + key_loss
        return total_loss, mouse_loss, key_loss

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Trainer initialization
    trainer = DTTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=cfg.batch_size,
        get_batch=lambda batch_size: next(iter(data_loader)),
        loss_fn=loss_fn,
        chunk_size=cfg.chunk_size,
        data_loader=data_loader
    )

    # Training loop
    best_loss = float('inf')
    for epoch in range(cfg.num_epochs):
        avg_loss = trainer.train_epoch(epoch)
        
        # Save model if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_decision_transformer_model.pth')
            print(f"New best model saved with loss: {best_loss}")

    # Save the final trained model
    torch.save(model.state_dict(), 'final_decision_transformer_model.pth')

    print("Training completed!")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()

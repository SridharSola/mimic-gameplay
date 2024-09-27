import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from decision_transformer.models.decision_transformer import DecisionTransformer
from data.data_loader import get_dataset
from decision_transformer.training.trainer import Trainer
import wandb
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project=cfg.wandb.project, config=cfg)

    # Get dataset
    dataset = get_dataset(cfg.data_dir, cfg.sequence_length, cfg.chunk_size)
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

    # Loss function
    def loss_fn(mouse_preds, key_preds, mouse_targets, key_targets):
        mouse_loss = nn.MSELoss()(mouse_preds, mouse_targets)
        key_loss = nn.BCEWithLogitsLoss()(key_preds, key_targets)
        return mouse_loss + key_loss

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Trainer initialization
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=cfg.batch_size,
        get_batch=lambda batch_size: next(iter(data_loader)),
        loss_fn=loss_fn,
        chunk_size=cfg.chunk_size
    )

    # Training loop
    best_loss = float('inf')
    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
        logs = trainer.train_iteration(num_steps=len(data_loader), print_logs=True)
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": logs['train/loss'],
            "train_mouse_loss": logs['train/mouse_loss'],
            "train_key_loss": logs['train/key_loss']
        })
        
        # Save model if loss improved
        if logs['train/loss'] < best_loss:
            best_loss = logs['train/loss']
            torch.save(model.state_dict(), 'best_decision_transformer_model.pth')
            print(f"New best model saved with loss: {best_loss}")

    # Save the final trained model
    torch.save(model.state_dict(), 'final_decision_transformer_model.pth')

    print("Training completed!")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()

import numpy as np
import torch
from tqdm import tqdm
import wandb

from decision_transformer.training.trainer import Trainer


class DTTrainer(Trainer):

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, chunk_size, data_loader, device):
        super().__init__(model, optimizer, batch_size, get_batch, loss_fn)
        self.chunk_size = chunk_size
        self.data_loader = data_loader
        self.device = device  # Add device attribute

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch + 1}")):
            self.optimizer.zero_grad()

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            states = batch['input_states']
            mouse_actions = batch['input_mouse_actions']
            key_actions = batch['input_key_actions']
            mouse_targets = batch['target_mouse_actions']
            key_targets = batch['target_key_actions']
            timesteps = batch['input_timesteps']

            mouse_preds, key_preds = self.model.forward(
                states, mouse_actions, key_actions, timesteps
            )

            loss, mouse_loss, key_loss = self.loss_fn(mouse_preds, key_preds, mouse_targets, key_targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log detailed information for each step
            wandb.log({
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
                "step_loss": loss.item(),
                "step_mouse_loss": mouse_loss.item(),
                "step_key_loss": key_loss.item()
            })

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch + 1,
            "epoch_avg_loss": avg_loss,
        })

        return avg_loss
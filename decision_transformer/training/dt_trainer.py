import numpy as np
import torch
from tqdm import tqdm
import wandb

from decision_transformer.training.trainer import Trainer


class DTTrainer(Trainer):

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, chunk_size, data_loader):
        super().__init__(model, optimizer, batch_size, get_batch, loss_fn)
        self.chunk_size = chunk_size
        self.data_loader = data_loader
        self.step = 0

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_mouse_loss = 0
        epoch_key_loss = 0
        num_batches = 0

        for data in tqdm(self.data_loader, desc=f"Epoch {epoch+1}"):
            states = data['input_states']
            mouse_actions = data['input_mouse_actions']
            key_actions = data['input_key_actions']
            mouse_action_target = data['target_mouse_actions']
            key_action_target = data['target_key_actions']
            timesteps = data['input_timesteps']

            # Add the last input action to the beginning of the target actions
            mouse_action_target = torch.cat([mouse_actions[:, -1:], mouse_action_target[:, :-1]], dim=1)
            key_action_target = torch.cat([key_actions[:, -1:], key_action_target[:, :-1]], dim=1)

            mouse_preds, key_preds = self.model.forward(
                states, mouse_actions, key_actions, timesteps
            )

            total_loss, mouse_loss, key_loss = self.loss_fn(
                mouse_preds, key_preds,
                mouse_action_target, key_action_target,
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            epoch_loss += total_loss.item()
            epoch_mouse_loss += mouse_loss.item()
            epoch_key_loss += key_loss.item()
            num_batches += 1

            # Log step losses
            wandb.log({
                'step': self.step,
                'step/loss': total_loss.item(),
                'step/mouse_loss': mouse_loss.item(),
                'step/key_loss': key_loss.item()
            })
            self.step += 1

        # Log epoch losses
        wandb.log({
            'epoch': epoch,
            'train/loss': epoch_loss / num_batches,
            'train/mouse_loss': epoch_mouse_loss / num_batches,
            'train/key_loss': epoch_key_loss / num_batches
        })

        return epoch_loss / num_batches  # Return average loss for the epoch
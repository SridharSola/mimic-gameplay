import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class DTTrainer(Trainer):

    def train_step(self):
        states, mouse_actions, key_actions, rtg, attention_mask = self.get_batch(self.batch_size)
        mouse_action_target, key_action_target = torch.clone(mouse_actions), torch.clone(key_actions)

        mouse_preds, key_preds = self.model.forward(
            states, mouse_actions, key_actions, attention_mask=attention_mask, target_return=rtg[:,0],
        )

        mouse_dim = mouse_preds.shape[2]
        key_dim = key_preds.shape[2]
        mouse_preds = mouse_preds.reshape(-1, mouse_dim)
        key_preds = key_preds.reshape(-1, key_dim)
        mouse_action_target = mouse_action_target[:,-1].reshape(-1, mouse_dim)
        key_action_target = key_action_target[:,-1].reshape(-1, key_dim)

        loss = self.loss_fn(
            mouse_preds, key_preds,
            mouse_action_target, key_action_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
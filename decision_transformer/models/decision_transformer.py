import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model

class DataStats:
    def __init__(self):
        self.feature_stats = {}

    def update(self, feature_stats):
        self.feature_stats = feature_stats

    def normalize_feature(self, value, key):
        if key not in self.feature_stats:
            return value
        min_val = self.feature_stats[key]['min']
        max_val = self.feature_stats[key]['max']
        if min_val == max_val:
            return torch.tensor(0.0)
        normalized = (2 * (value - min_val) / (max_val - min_val)) - 1
        return torch.clamp(normalized, -1, 1)

class DecisionTransformer(TrajectoryModel):
    def __init__(
            self,
            state_dim,
            mouse_dim,
            key_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            n_layer=4,
            n_head=4,
            n_inner=None,
            activation_function="relu",
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            chunk_size=5,
            **kwargs
    ):
        super().__init__(state_dim, mouse_dim, key_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.mouse_dim = mouse_dim
        self.key_dim = key_dim
        self.data_stats = DataStats()
        self.chunk_size = chunk_size

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            n_positions=n_positions,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_ctx=n_positions,
            **kwargs
        )

        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_mouse = torch.nn.Linear(self.mouse_dim, hidden_size)
        self.embed_key = torch.nn.Linear(self.key_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_mouse = nn.Sequential(
            *([nn.Linear(hidden_size, self.mouse_dim * self.chunk_size)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_key = nn.Linear(hidden_size, self.key_dim * self.chunk_size)

    def set_data_stats(self, feature_stats):
        self.data_stats.update(feature_stats)

    def forward(self, states, mouse_actions, key_actions, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        mouse_embeddings = self.embed_mouse(mouse_actions)
        key_embeddings = self.embed_key(key_actions)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        mouse_embeddings = mouse_embeddings + time_embeddings[:, :-1]  # One less action
        key_embeddings = key_embeddings + time_embeddings[:, :-1]  # One less action

        # Stack inputs: (s_{t-k}, a_{t-k}, s_{t-k+1}, a_{t-k+1}, ..., s_{t-1}, a_{t-1}, s_{t})
        stacked_inputs = torch.stack(
            (state_embeddings[:, :-1], mouse_embeddings, key_embeddings, state_embeddings[:, 1:]), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 4*(seq_length-1), self.hidden_size)
        
        # Add the final state embedding (s_{t}) to the sequence
        stacked_inputs = torch.cat([stacked_inputs, state_embeddings[:, -1].unsqueeze(1)], dim=1)
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Adjust attention mask to include the final state
        stacked_attention_mask = torch.stack(
            (attention_mask[:, :-1], attention_mask[:, :-1], attention_mask[:, :-1], attention_mask[:, 1:]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 4*(seq_length-1))
        stacked_attention_mask = torch.cat([stacked_attention_mask, attention_mask[:, -1].unsqueeze(1)], dim=1)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # Use the last state embedding to predict the next chunk of actions
        last_state_embed = x[:, -1]

        mouse_preds = self.predict_mouse(last_state_embed).reshape(batch_size, self.chunk_size, self.mouse_dim)
        key_preds = self.predict_key(last_state_embed).reshape(batch_size, self.chunk_size, self.key_dim)

        return mouse_preds, key_preds

    def get_action(self, states, mouse_actions, key_actions, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        mouse_actions = mouse_actions.reshape(1, -1, self.mouse_dim)
        key_actions = key_actions.reshape(1, -1, self.key_dim)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            mouse_actions = mouse_actions[:,-self.max_length:]
            key_actions = key_actions[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            mouse_actions = torch.cat(
                [torch.zeros((mouse_actions.shape[0], self.max_length - mouse_actions.shape[1], self.mouse_dim),
                             device=mouse_actions.device), mouse_actions],
                dim=1).to(dtype=torch.float32)
            key_actions = torch.cat(
                [torch.zeros((key_actions.shape[0], self.max_length - key_actions.shape[1], self.key_dim),
                             device=key_actions.device), key_actions],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        mouse_preds, key_preds = self.forward(
            states, mouse_actions, key_actions, timesteps, attention_mask=attention_mask, **kwargs)

        return mouse_preds[0], key_preds[0]
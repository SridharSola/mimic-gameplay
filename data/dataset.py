import torch
from torch.utils.data import Dataset
from safetensors import safe_open
import os
import numpy as np

class SafetensorsDataset(Dataset):
    def __init__(self, data_dir, sequence_length, chunk_size=5):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.safetensors')]
        self.file_sizes = []
        self.cumulative_sizes = [0]
        self.feature_stats = {}
        self.all_keys = set()  # Initialize an empty set for all keys
        self.special_values = {
            'EnemyDistance': -100000,
            'RelativeYaw': -360,
            # Add other features with special values here
        }
        self.na_value = -2  # Our chosen value to represent NA

        # Calculate file sizes, feature stats, and gather all keys
        for file in self.files:
            file_path = os.path.join(data_dir, file)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                file_size = len(f.get_tensor('Time(s)'))
                self.file_sizes.append(file_size)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + max(0, file_size - self.sequence_length - self.chunk_size + 2))
                
                # Calculate feature stats and gather all keys
                for key in f.keys():
                    if key == 'PressedKeys':
                        # Gather all unique keys from this file
                        pressed_keys = f.get_tensor(key)
                        for pk in pressed_keys:
                            self.all_keys.update(pk.split('|'))
                    elif key != 'Time(s)':  # Skip time as we're not using it
                        tensor = f.get_tensor(key)
                        if key in self.special_values:
                            # Filter out special values for min/max calculation
                            valid_values = tensor[tensor != self.special_values[key]]
                        else:
                            valid_values = tensor

                        if len(valid_values) > 0:
                            min_val = valid_values.min().item()
                            max_val = valid_values.max().item()
                        else:
                            min_val = max_val = 0  # or another appropriate default

                        if key not in self.feature_stats:
                            self.feature_stats[key] = {'min': min_val, 'max': max_val}
                        else:
                            self.feature_stats[key]['min'] = min(self.feature_stats[key]['min'], min_val)
                            self.feature_stats[key]['max'] = max(self.feature_stats[key]['max'], max_val)

        # Create a mapping from keys to indices
        self.key_to_index = {key: i for i, key in enumerate(sorted(self.all_keys))}
        self.key_dim = len(self.all_keys)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Find which file this index belongs to
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        
        # Calculate the starting position within the file
        start_idx = idx - self.cumulative_sizes[file_idx]
        
        file_path = os.path.join(self.data_dir, self.files[file_idx])
        with safe_open(file_path, framework="pt", device="cpu") as f:
            states, mouse_actions, key_actions, timesteps = [], [], [], []
            for i in range(self.sequence_length + self.chunk_size):
                state, mouse_action, key_action = self.process_row(f, start_idx + i)
                states.append(state)
                mouse_actions.append(mouse_action)
                key_actions.append(key_action)
                timesteps.append(i)  # Relative timestep
            
        states = torch.stack(states)
        mouse_actions = torch.stack(mouse_actions)
        key_actions = torch.stack(key_actions)
        timesteps = torch.tensor(timesteps, dtype=torch.long)

        # Input sequence
        input_states = states[:self.sequence_length]
        input_mouse_actions = mouse_actions[:self.sequence_length]
        input_key_actions = key_actions[:self.sequence_length]
        input_timesteps = timesteps[:self.sequence_length]

        # Target action chunks
        target_mouse_actions = mouse_actions[self.sequence_length:].reshape(-1, self.chunk_size, 2)
        target_key_actions = key_actions[self.sequence_length:].reshape(-1, self.chunk_size, self.key_dim)

        return {
            'input_states': input_states,
            'input_mouse_actions': input_mouse_actions,
            'input_key_actions': input_key_actions,
            'input_timesteps': input_timesteps,
            'target_mouse_actions': target_mouse_actions,
            'target_key_actions': target_key_actions
        }

    def process_row(self, f, row_idx):
        state_tensors = []
        mouse_action = None

        for key in f.keys():
            if key == 'Time(s)':
                continue  # Skip the absolute time
            elif key == 'PressedKeys':
                key_action_vector = self.process_key_action(f.get_tensor(key)[row_idx])
            elif key in ['MouseDeltaX(pixels)', 'MouseDeltaY(pixels)']:
                normalized_value = self.normalize_feature(f.get_tensor(key)[row_idx], key)
                if mouse_action is None:
                    mouse_action = normalized_value.unsqueeze(0)
                else:
                    mouse_action = torch.cat([mouse_action, normalized_value.unsqueeze(0)])
            else:  # All other features are part of the state
                normalized_value = self.normalize_feature(f.get_tensor(key)[row_idx], key)
                state_tensors.append(normalized_value.unsqueeze(0))

        state = torch.cat(state_tensors)
        
        return state, mouse_action, key_action_vector

    def normalize_feature(self, value, key):
        min_val = self.feature_stats[key]['min']
        max_val = self.feature_stats[key]['max']

        # Check if this feature has a special value
        if key in self.special_values and value == self.special_values[key]:
            return torch.tensor(self.na_value)

        # Handle constant features
        if min_val == max_val:
            return torch.tensor(0.0)

        # Normal normalization for non-special values
        normalized = (2 * (value - min_val) / (max_val - min_val)) - 1
        return torch.clamp(normalized, -1, 1)

    def process_key_action(self, pressed_keys):
        # Convert the pressed keys string to a set of individual keys
        keys_pressed = set(pressed_keys.split('|'))
        
        # Create a one-hot encoding
        one_hot = torch.zeros(self.key_dim)
        for key in keys_pressed:
            if key in self.key_to_index:
                one_hot[self.key_to_index[key]] = 1
        
        return one_hot

    def get_key_dim(self):
        return self.key_dim

    def get_data_stats(self):
        return {
            'feature_stats': self.feature_stats,
            'key_to_index': self.key_to_index,
            'key_dim': self.key_dim
        }

def get_dataset(data_dir, sequence_length, chunk_size=5):
    return SafetensorsDataset(data_dir, sequence_length, chunk_size)

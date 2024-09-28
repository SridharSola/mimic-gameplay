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
        self.special_values = {
            'Distance': -1000000,
            'RelativePosition': 0.00,
            'RelativeYaw': -360,
        }
        self.na_value = -2  # Our chosen value to represent NA

        # Calculate file sizes and feature stats
        for file in self.files:
            file_path = os.path.join(data_dir, file)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                file_size = len(f.get_tensor('Time(s)'))
                self.file_sizes.append(file_size)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + file_size)
                
                # Calculate feature stats
                for key in f.keys():
                    if key != 'Time(s)' and key != 'PressedKeys' and not key.endswith('_mapping'):
                        tensor = f.get_tensor(key)
                        special_value = self.get_special_value(key)
                        if special_value is not None:
                            valid_values = tensor[tensor != special_value]
                        else:
                            valid_values = tensor

                        if len(valid_values) > 0:
                            min_val = valid_values.min().item()
                            max_val = valid_values.max().item()
                        else:
                            min_val = max_val = 0

                        if key not in self.feature_stats:
                            self.feature_stats[key] = {'min': min_val, 'max': max_val}
                        else:
                            self.feature_stats[key]['min'] = min(self.feature_stats[key]['min'], min_val)
                            self.feature_stats[key]['max'] = max(self.feature_stats[key]['max'], max_val)
                        
                        # Print key mapping for each file
                        key_mapping = self.decode_string_array(f.get_tensor('PressedKeys_mapping'))
                        # print(f"File: {file}, Key mapping: {key_mapping}")
                        # print(f"Key dimension: {f.get_tensor('PressedKeys').shape[1]}")


        # Get the key dimension and mapping from the first file
        with safe_open(os.path.join(data_dir, self.files[0]), framework="pt", device="cpu") as f:
            self.key_dim = f.get_tensor('PressedKeys').shape[1]
            self.key_mapping = self.decode_string_array(f.get_tensor('PressedKeys_mapping'))

        print(f"Dataset initialized with {len(self.files)} files, total size: {self.cumulative_sizes[-1]}")

    def get_special_value(self, key):
        for special_key, value in self.special_values.items():
            if special_key in key:
                return value
        return None

    def __len__(self):
        return self.cumulative_sizes[-1] - self.sequence_length - self.chunk_size + 1

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        start_idx = idx - self.cumulative_sizes[file_idx]
        
        file_path = os.path.join(self.data_dir, self.files[file_idx])
        with safe_open(file_path, framework="pt", device="cpu") as f:
            sequence_length = self.sequence_length + self.chunk_size
            states, mouse_actions, key_actions = [], [], []
            
            for key in f.keys():
                if key == 'Time(s)' or key.endswith('_mapping'):
                    continue
                tensor = f.get_tensor(key)[start_idx:start_idx+sequence_length]
                if key == 'PressedKeys':
                    key_actions = tensor
                    # print(f"Original key_actions shape: {key_actions.shape}")
                elif key in ['MouseDeltaX(pixels)', 'MouseDeltaY(pixels)']:
                    mouse_actions.append(self.normalize_feature(tensor, key))
                else:
                    states.append(self.normalize_feature(tensor, key))
        
        states = torch.stack(states, dim=1)
        mouse_actions = torch.stack(mouse_actions, dim=1)

        # Ensure we have enough data for both input and target
        total_length = self.sequence_length + self.chunk_size
        if states.shape[0] < total_length:
            # Pad if necessary
            pad_length = total_length - states.shape[0]
            states = torch.nn.functional.pad(states, (0, 0, 0, pad_length))
            mouse_actions = torch.nn.functional.pad(mouse_actions, (0, 0, 0, pad_length))
            key_actions = torch.nn.functional.pad(key_actions, (0, 0, 0, pad_length))

        # Input sequence
        input_states = states[:self.sequence_length]
        input_mouse_actions = mouse_actions[:self.sequence_length-1]  # Up to action at t-1
        input_key_actions = key_actions[:self.sequence_length-1]  # Up to action at t-1
        input_timesteps = torch.arange(self.sequence_length)

        # Target action chunks
        target_mouse_actions = mouse_actions[self.sequence_length-1:self.sequence_length-1+self.chunk_size]
        target_key_actions = key_actions[self.sequence_length-1:self.sequence_length-1+self.chunk_size]

        return {
            'input_states': input_states,
            'input_mouse_actions': input_mouse_actions,
            'input_key_actions': input_key_actions,
            'input_timesteps': input_timesteps,
            'target_mouse_actions': target_mouse_actions,
            'target_key_actions': target_key_actions
        }

    def normalize_feature(self, tensor, key):
        special_value = self.get_special_value(key)
        if special_value is not None:
            # Replace special values with NA value
            tensor = torch.where(tensor == special_value, torch.tensor(self.na_value, dtype=tensor.dtype), tensor)
        
        min_val = self.feature_stats[key]['min']
        max_val = self.feature_stats[key]['max']
        
        if min_val == max_val:
            return torch.zeros_like(tensor)
        
        # Normalize the non-NA values
        normalized_tensor = (2 * (tensor - min_val) / (max_val - min_val)) - 1
        
        # Keep NA values as they are (don't normalize them)
        normalized_tensor = torch.where(tensor == self.na_value, torch.tensor(self.na_value, dtype=normalized_tensor.dtype), normalized_tensor)
        
        return normalized_tensor

    @staticmethod
    def decode_string_array(int_array):
        return ''.join(chr(i) for i in int_array if i != 0).split('|')

    def get_data_stats(self):
        return {
            'feature_stats': self.feature_stats,
            'key_dim': self.key_dim,
            'key_mapping': self.key_mapping
        }

    def get_key_dim(self):
        """
        Returns the dimension of the key action vector.
        """
        return self.key_dim

def get_dataset(data_dir, sequence_length, chunk_size=5):
    return SafetensorsDataset(data_dir, sequence_length, chunk_size)
import argparse
from safetensors import safe_open
import numpy as np

def decode_string_array(int_array):
    return ''.join(chr(i) for i in int_array if i != 0).split('|')

def format_value(value):
    if isinstance(value, np.ndarray):
        return ' '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in value)
    elif isinstance(value, float):
        return f'{value:.4f}'
    return str(value)

def print_safetensor_range(file_path, start_index, end_index):
    with safe_open(file_path, framework="np") as f:
        tensor_names = [name for name in f.keys() if not name.endswith("_mapping")]
        
        # Decode mapping
        pressed_keys_mapping = decode_string_array(f.get_tensor("PressedKeys_mapping"))

        max_name_length = max(len(name) for name in tensor_names)

        for i in range(start_index, min(end_index, min(f.get_tensor(name).shape[0] for name in tensor_names))):
            print(f"Index: {i}")
            print("-" * 40)
            for name in tensor_names:
                tensor = f.get_tensor(name)
                if name == "PressedKeys":
                    pressed = [pressed_keys_mapping[j] for j, is_pressed in enumerate(tensor[i]) if is_pressed]
                    value = "|".join(pressed) if pressed else "None"
                    print(f"{name:<{max_name_length}}: {value}")
                    print(f"{'PressedKeys (raw)':<{max_name_length}}: {tensor[i]}")
                    print(f"{'Key positions':<{max_name_length}}: {pressed_keys_mapping}")
                else:
                    value = format_value(tensor[i])
                    print(f"{name:<{max_name_length}}: {value}")
            print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a range of values from a safetensor file.")
    parser.add_argument("file_path", type=str, help="Path to the safetensor file")
    parser.add_argument("start_index", type=int, help="Start index of the range to print")
    parser.add_argument("end_index", type=int, help="End index of the range to print")

    args = parser.parse_args()

    print_safetensor_range(args.file_path, args.start_index, args.end_index)

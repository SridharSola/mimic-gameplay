import csv
import sys
import os
import glob
import numpy as np
from safetensors.numpy import save_file
import json

def convert_csv_to_safetensor(input_file, output_folder):
    # Read CSV file
    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Get all unique keys, excluding 'RightMouseButton'
    all_keys = set()
    for row in data:
        keys = row['PressedKeys'].split('|')
        all_keys.update(key for key in keys if key and key != 'RightMouseButton')

    valid_keys = sorted(list(all_keys))
    print(f"Valid keys: {valid_keys}")
    print(f"Removed key: RightMouseButton")

    # Create tensor dictionary
    tensor_dict = {}
    for key in reader.fieldnames:
        if key == 'PressedKeys':
            # For pressed keys, create a one-hot encoded tensor
            one_hot = np.zeros((len(data), len(valid_keys)), dtype=np.float32)
            for i, row in enumerate(data):
                for k in row[key].split('|'):
                    if k in valid_keys:
                        one_hot[i, valid_keys.index(k)] = 1
            tensor_dict[key] = one_hot
            # Store the key mapping
            tensor_dict[f"{key}_mapping"] = np.array([ord(c) for c in '|'.join(valid_keys)], dtype=np.int32)
        elif key in ['MouseX(0-1)', 'MouseY(0-1)']:
            # For mouse position, convert to float
            tensor_dict[key] = np.array([float(row[key]) for row in data], dtype=np.float32)
        else:
            # For all other numeric columns
            values = [float(row[key]) if row[key] != '' else np.nan for row in data]
            if all(np.isnan(values)):
                print(f"Skipping column '{key}' as it contains only NaN values")
                continue
            tensor_dict[key] = np.array(values, dtype=np.float32)

    # Remove any remaining columns with NaN values
    tensor_dict = {k: v for k, v in tensor_dict.items() if not np.isnan(v).any()}

    # Save as safetensor
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + '.safetensors')
    save_file(tensor_dict, output_file)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convertcsv.py <input_folder_or_pattern> <output_folder>")
        sys.exit(1)

    input_pattern = sys.argv[1]
    output_folder = sys.argv[2]

    # Get list of input files
    input_files = glob.glob(input_pattern)

    if not input_files:
        print(f"Error: No files found matching {input_pattern}")
        sys.exit(1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_file in input_files:
        convert_csv_to_safetensor(input_file, output_folder)

    print(f"Converted {len(input_files)} files.")

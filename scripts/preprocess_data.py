import csv
import sys
import os
import glob
import numpy as np
from safetensors.numpy import save_file

def convert_csv_to_safetensor(input_file, output_folder):
    # Read CSV file
    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Convert data to numpy arrays
    tensor_dict = {}
    for key in data[0].keys():
        if key != 'PressedKeys' and key not in ['MouseX(0-1)', 'MouseY(0-1)', 'InputType']:
            tensor_dict[key] = np.array([float(row[key]) for row in data])
        elif key == 'PressedKeys':
            tensor_dict[key] = np.array([row[key] for row in data])

    # Create output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_folder, f"{base_name}.safetensors")

    # Save as SafeTensor
    save_file(tensor_dict, output_file)
    print(f"Converted {input_file} to {output_file}")

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

# read jsonl files
import json
import os
from sklearn.model_selection import train_test_split

def read_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def load_data(model_name, data_name, split = 'train', results_path='results'):
    """Load data from the specified path."""
    path = f"{results_path}/{model_name}/{data_name}_results.jsonl"
    if not os.path.exists(path):
        print(f"Error: Data file not found at {path}")
        return None
    data = read_jsonl(path)

    if split in ['train', 'test']:
        train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)
        return train_data if split == 'train' else test_data
    elif split == 'full':
        return data
    else:
        raise ValueError(f"Unknown split: {split}")
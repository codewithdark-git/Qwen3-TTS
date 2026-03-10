# Prepare Data for Hugging Face Datasets

import pandas as pd
from datasets import load_dataset, Dataset


def prepare_data(file_path: str, split: str = 'train') -> Dataset:
    """Load and prepare dataset for Hugging Face.

    Args:
        file_path (str): Path to the data file.
        split (str): Split of the dataset to load (e.g., 'train', 'test').

    Returns:
        Dataset: A Hugging Face Dataset object ready for training.
    """
    # Load data from the provided file path
    data = pd.read_csv(file_path)

    # Create a Hugging Face Dataset from the DataFrame
    dataset = Dataset.from_pandas(data)

    # Optionally splitting the dataset
dataset = dataset.train_test_split(test_size=0.1)

    return dataset[split]  


# Example usage
# dataset = prepare_data('data/my_dataset.csv')
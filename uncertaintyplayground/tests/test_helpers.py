import numpy as np
from utils.generate_data import generate_multi_modal_data

def generate_test_data(num_samples, modes):
    """Generates test data.

    Args:
        num_samples (int): The number of samples to generate.

    Returns:
        tuple: A tuple containing X (np.ndarray) and y (np.ndarray).
    """
    np.random.seed(42)
    X = np.random.rand(num_samples, 20)
    y = generate_multi_modal_data(num_samples, modes)
    return X, y

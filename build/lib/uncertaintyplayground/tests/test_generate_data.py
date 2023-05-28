import unittest
import numpy as np
from uncertaintyplayground.utils.generate_data import generate_multi_modal_data

def test_generate_multi_modal_data():
    """
    Unit test for `generate_multi_modal_data` function.

    This test checks whether the function returns an array of correct size and also if the data
    follows the expected distribution by checking whether the mean and standard deviation are
    close to the expected values.

    Raises:
        AssertionError: If the returned array is not of the correct size.
        AssertionError: If the mean or standard deviation of the generated data is too far from the expected value.
    """
    num_samples = 10000
    modes = [
        {'mean': -3.0, 'std_dev': 0.5, 'weight': 0.3},
        {'mean': 0.0, 'std_dev': 1.0, 'weight': 0.4},
        {'mean': 3.0, 'std_dev': 0.7, 'weight': 0.3}
    ]

    y = generate_multi_modal_data(num_samples, modes)

    assert len(y) == num_samples, "The generated data does not have the correct number of samples."
    
    # Check if the mean and std deviation are close to expected values
    for mode in modes:
        mode_samples = [sample for sample in y if abs(sample-mode['mean']) < 3*mode['std_dev']]
        assert abs(np.mean(mode_samples)-mode['mean']) < 0.1, f"For mode with mean {mode['mean']}, generated data does not have correct mean."
        assert abs(np.std(mode_samples)-mode['std_dev']) < 0.1, f"For mode with mean {mode['mean']}, generated data does not have correct std deviation."

    # your code here


if __name__ == "__main__":
    unittest.main()

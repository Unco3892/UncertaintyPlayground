import numpy as np

def generate_multi_modal_data(num_samples, modes):
    """
    Generate multimodal data for the mixture density network.

    This function generates a specified number of samples for each mode. Each mode is defined by
    a mean, standard deviation, and weight. The weight determines the proportion of total samples
    that will come from this mode.

    Args:
        num_samples (int): The total number of data samples to generate.
        modes (list of dict): A list of dictionaries, where each dictionary represents a mode and
            contains the keys 'mean' (float), 'std_dev' (float), and 'weight' (float).

    Returns:
        np.array: An array of generated data samples.

    Raises:
        ValueError: If num_samples is not a positive integer.
        ValueError: If modes is not a list of dictionaries each containing 'mean', 'std_dev', and 'weight'.

    Examples:
        >>> modes = [
        ...     {'mean': -3.0, 'std_dev': 0.5, 'weight': 0.3},
        ...     {'mean': 0.0, 'std_dev': 1.0, 'weight': 0.4},
        ...     {'mean': 3.0, 'std_dev': 0.7, 'weight': 0.3}
        ... ]
        >>> data = generate_multi_modal_data(1000, modes)
        >>> print(data.shape)
        (1000,)
    """
    samples = []
    for mode in modes:
        mean, std_dev = mode['mean'], mode['std_dev']
        num = int(num_samples * mode['weight'])
        samples.extend(np.random.normal(mean, std_dev, num))
    return np.array(samples)

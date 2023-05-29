import matplotlib.pyplot as plt
import numpy as np

def plot_results_grid(trainer, compare_func, X_test, indices, Y_test= None, ncols=2, dtype=np.float32):
    """
    Plot a grid of comparison plots (minimum 2) for a set of test instances.

    Args:
        trainer (object): The trained MDNTrainer or SparseGPTrainer instance.
        compare_func (function): Function to compare distributions (compare_distributions for MDN or compare_distributions_svgpr for SVGPR).
        X_test (np.ndarray): The test input data of shape (num_samples, num_features).
        indices (list): The indices of the instances to plot.
        ncols (int, optional): Number of columns in the grid. Default is 2.
        Y_test (np.ndarray): The test target data of shape (num_samples,). Default is None.
        dtype (np.dtype, optional): Data type to use for plotting. Default is np.float32.

    Returns:
        None
    """
    num_instances = len(indices)
    nrows = (num_instances - 1) // ncols + 1

    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    for i, ax in zip(indices, axes.flat):
        x_instance = X_test[i].astype(dtype)
        if Y_test is None:
            y_actual = None
        else:
            y_actual = Y_test[i].astype(dtype)
        compare_func(trainer, x_instance, y_actual, ax=ax)
        ax.set_title(f"Test Instance: {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(axis='y', alpha=0.75)

    # Remove empty subplots
    if num_instances < nrows * ncols:
        for ax in axes.flat[num_instances:]:
            ax.remove()

    plt.tight_layout()
    plt.show()

# Designig a context manager to disable the display of matplotlib plots when testing the plots
class DisablePlotDisplay:
    """
    Context manager to disable the display of matplotlib plots.
    """

    def __enter__(self):
        plt.ioff()  # Turn off interactive mode
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.close()  # Close the plot


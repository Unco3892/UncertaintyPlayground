import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def compare_distributions_svgpr(trainer, x_instance, y_actual=None, num_samples=10000, ax=None):
    """
    Compare the actual and predicted outcome distributions.

    Args:
        trainer (SVGPRTrainer): The trained SVGPRTrainer instance.
        x_instance (np.ndarray): The instance for which to predict the outcome distribution.
        y_actual (float or np.ndarray, optional): The actual outcome. If a single value, plot as a vertical line.
                                                  If an array or list, plot as a KDE. If None, don't plot actual outcome.
        num_samples (int, optional): The number of samples to generate from the predicted distribution.
                                     Default is 10000.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None, create a new figure.

    Returns:
        None
    """
    # Ensure x_instance is a 2D array
    if x_instance.ndim == 1:
        x_instance = np.expand_dims(x_instance, axis=0)

    # Get the predicted mean and standard deviation
    mu, sigma = trainer.predict_with_uncertainty(x_instance)

    # Generate samples from the predicted distribution
    predicted_samples = np.random.normal(mu, sigma, num_samples)

    # Plot KDE of predicted samples
    if ax is None:
        sns.kdeplot(predicted_samples, fill=True, color="r", label="Predicted distribution")
        plt.axvline(mu, color="r", linestyle="--", label="Predicted value")
    else:
        sns.kdeplot(predicted_samples, fill=True, color="r", label="Predicted distribution", ax=ax)
        ax.axvline(mu, color="r", linestyle="--", label="Predicted value")

    # Plot the actual value
    if y_actual is not None:
        if ax is None:
            plt.axvline(y_actual, color="b", linestyle="--", label="Actual value")
        else:
            ax.axvline(y_actual, color="b", linestyle="--", label="Actual value")
    
    # Set an option when doing multiple plots
    if ax is None:
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.show()


# def plot_results_grid_svgpr(trainer, X_test, Y_test, indices, ncols=2, dtype=np.float32):
#     """
#     Plot a grid of comparison plots (minimum 2) for a set of test instances.

#     Args:
#         trainer (SVGPRTrainer): The trained SVGPRTrainer instance.
#         X_test (np.ndarray): The test input data of shape (num_samples, num_features).
#         Y_test (np.ndarray): The test target data of shape (num_samples,).
#         indices (list): The indices of the instances to plot.
#         ncols (int, optional): Number of columns in the grid. Default is 2.
#         dtype (np.dtype, optional): Data type to use for plotting. Default is np.float32.

#     Returns:
#         None
#     """
#     num_instances = len(indices)
#     nrows = (num_instances - 1) // ncols + 1

#     _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

#     for i, ax in zip(indices, axes.flat):
#         x_instance = X_test[i].astype(dtype)
#         y_actual = Y_test[i].astype(dtype)
#         compare_distributions_svgpr(trainer, x_instance, y_actual, ax=ax)
#         ax.set_title(f"Test Instance: {i}")
#         ax.set_xlabel("Value")
#         ax.set_ylabel("Density")
#         ax.legend()
#         ax.grid(axis='y', alpha=0.75)

#     # Remove empty subplots
#     if num_instances < nrows * ncols:
#         for ax in axes.flat[num_instances:]:
#             ax.remove()

#     plt.tight_layout()
#     plt.show()


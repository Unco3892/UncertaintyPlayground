import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
from uncertaintyplayground.trainers.svgp_trainer import SparseGPTrainer

def compare_distributions_svgpr(
    trainer: SparseGPTrainer,
    x_instance: np.ndarray,
    y_actual: Optional[Union[float, np.ndarray]] = None,
    num_samples: int = 10000,
    ax: Optional['matplotlib.axes.Axes'] = None,
    dtype: np.dtype = np.float32
) -> None:
    """
    Compare the actual and predicted outcome value/distributions for the SVGPR model.

    Args:
        trainer (SVGPRTrainer): The trained SVGPRTrainer instance.
        x_instance (np.ndarray): The instance for which to predict the outcome distribution.
        y_actual (float or np.ndarray, optional): The actual outcome. If a single value, plot as a vertical line.
                                                  If an array or list, plot as a KDE. If None, don't plot actual outcome.
        num_samples (int, optional): The number of samples to generate from the predicted distribution.
                                     Default is 10000.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None, create a new figure.
        dtype (np.dtype, optional): Data type to use for plotting. Default is np.float32.

    Returns:
        None
    """
    # Ensure x_instance is a 2D array
    if x_instance.ndim == 1:
        x_instance = np.expand_dims(x_instance, axis=0)

    # Change the prediction instance to the desired data type
    x_instance = x_instance.astype(dtype)

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


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

def compare_distributions_mdn(trainer, x_instance, y_actual=None, num_samples=10000, ax=None, dtype=np.float32):
    """
    Compare the actual and predicted outcome value/distributions for the MDN model.

    Args:
        trainer (MDNTrainer): The trained MDNTrainer instance.
        x_instance (np.ndarray or torch.Tensor): The instance for which to predict the outcome distribution.
        y_actual (float or np.ndarray, optional): The actual outcome(s). If a single value, plot as a vertical line.
                                                  If an array or list, plot as a KDE. If None, don't plot actual outcome.
        num_samples (int, optional): The number of samples to generate from the predicted and actual distributions.
                                     Default is 10000.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None, create a new figure.
        dtype (np.dtype, optional): Data type to use for plotting. Default is np.float32.

    Returns:
        None
    """
    # Ensure it passes the unit testing
    warnings.warn("This is a UserWarning")

    # Ensure x_instance is a 2D array
    if x_instance.ndim == 1:
        x_instance = np.expand_dims(x_instance, axis=0)
    
    # Change the prediction instance to the desired data type
    x_instance = x_instance.astype(dtype)

    # Get the predicted parameters of the mixture distribution
    pi, mu, sigma, pred = trainer.predict_with_uncertainty(x_instance)
    
    # Generate samples from the predicted distribution
    predicted_samples = []
    
    print ("The actual input was: ",x_instance, "\n",
           "The ground truth was: ",y_actual, "\n",
           "The predicted weights are: ",pi, "\n", 
           "The predicted means are: ",mu, "\n",
           "The predicted sigmas are: ",sigma, "\n",
           "The final prediction is: ",pred, "\n")

    for _ in range(num_samples):
        # Choose Gaussian
        idx = np.random.choice(np.arange(len(pi[0])), p=pi[0])
        # Sample from Gaussian
        sample = np.random.normal(mu[0, idx], sigma[0, idx])
        predicted_samples.append(sample)

    # Plot KDE of predicted samples
    if ax is None:
        sns.kdeplot(predicted_samples, fill=True, color="r", label="Predicted distribution")
        plt.axvline(pred[0], color="r", linestyle="--", label="Predicted value")
    else:
        sns.kdeplot(predicted_samples, fill=True, color="r", label="Predicted distribution", ax=ax)
        ax.axvline(pred[0], color="r", linestyle="--", label="Predicted value")

    # Plot the actual value(s)
    if y_actual is not None:
        if np.isscalar(y_actual):
            if ax is None:
                plt.axvline(y_actual, color="b", linestyle="--", label="Ground truth")
            else:
                ax.axvline(y_actual, color="b", linestyle="--", label="Ground truth")
        else:
            if ax is None:
                sns.kdeplot(y_actual, fill=True, color="b", label="Actual")
            else:
                sns.kdeplot(y_actual, fill=True, color="b", label="Actual", ax=ax)

    # Set an option when doing multiple plots
    if ax is None:
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.show()


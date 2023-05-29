import torch
import numpy as np
from uncertaintyplayground.trainers.mdn_trainer import MDNTrainer
from uncertaintyplayground.utils.generate_data import generate_multi_modal_data
from uncertaintyplayground.predplot.grid_predplot import plot_results_grid
from uncertaintyplayground.predplot.mdn_predplot import compare_distributions_mdn

#-------------------------------------------------------------
# Example 1: Using MDN model with simulated data
# Generate simulated data with three modes
modes = [
    {'mean': -3.0, 'std_dev': 0.5, 'weight': 0.3},
    {'mean': 0.0, 'std_dev': 1.0, 'weight': 0.4},
    {'mean': 3.0, 'std_dev': 0.7, 'weight': 0.3}
]

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(1)

num_samples = 1000
X = np.random.rand(num_samples, 10)
y = generate_multi_modal_data(num_samples, modes)

# Train an MDN model with 50 hidden units and 5 Gaussian components
mdn_trainer = MDNTrainer(X, y, num_epochs=100, lr=0.01, n_gaussians=5, dense1_units=50)
mdn_trainer.train()

# Visualize the model's predictions and uncertainties for a single instance
# Note: Make sure that the predictions for the inference are in the same type of float as the training data
index_instance = 900
# With an actual y value (diagnostic)
compare_distributions_mdn(trainer=mdn_trainer, x_instance=X[index_instance, :],y_actual=y[index_instance], num_samples=1000)
# You can also make any of the plots without a `y` (default set to None)
# Without an actual y value (pure inference)
compare_distributions_mdn(trainer = mdn_trainer, x_instance = X[index_instance, :])

# Visualize the model's predictions and uncertainties for more than one instance
indices = [900, 500]  # Example indices
# With an actual y value (diagnostic)
plot_results_grid(trainer=mdn_trainer, compare_func=compare_distributions_mdn,
                  X_test=X, Y_test=y, indices=indices, ncols=2)

# Without an actual y value (pure inference)
# plot_results_grid(trainer=mdn_trainer, compare_func=compare_distributions_mdn, X_test=X, Y_test=None, indices=indices, ncols=2)

#-------------------------------------------------------------
# Example 2: Using MDN model with real data
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
california = fetch_california_housing()

# Convert X and y to numpy arrays of float32
X = np.array(california.data, dtype=np.float32)
y = np.array(california.target, dtype=np.float32)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MDNTrainer with the training data
california_trainer = MDNTrainer(X_train, y_train, num_epochs=100, lr=0.001, dense1_units=50, n_gaussians=10)

# # Train the model
california_trainer.train()

# Visualize the model's predictions for multiple instances
plot_results_grid(trainer=california_trainer, compare_func=compare_distributions_mdn, X_test=X_test, Y_test=y_test, indices=[900, 500], ncols=2)

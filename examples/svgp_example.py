import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from uncertaintyplayground.trainers.svgp_trainer import SparseGPTrainer
from uncertaintyplayground.predplot.grid_predplot import plot_results_grid
from uncertaintyplayground.predplot.svgp_predplot import compare_distributions_svgpr

# Load the California Housing dataset
california = fetch_california_housing()

# Convert X and y to numpy arrays of float32
X = np.array(california.data, dtype=np.float32)
y = np.array(california.target, dtype=np.float32)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SGP model with 50 inducing points
california_trainer = SparseGPTrainer(X_train, y_train, num_inducing_points=100, num_epochs=30, batch_size=512, lr=0.1, patience=3)
california_trainer.train()

# Visualize the model's predictions for multiple instances
plot_results_grid(trainer=california_trainer, compare_func=compare_distributions_svgpr, X_test=X_test, Y_test=y_test, indices=[900, 500], ncols=2)

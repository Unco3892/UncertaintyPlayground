from uncertaintyplayground.trainers.svgp_trainer import SparseGPTrainer
from uncertaintyplayground.trainers.mdn_trainer import MDNTrainer
from uncertaintyplayground.predplot.svgp_predplot import compare_distributions_svgpr
from uncertaintyplayground.predplot.mdn_predplot import compare_distributions_mdn
from uncertaintyplayground.predplot.grid_predplot import plot_results_grid
import torch
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# SVGPR: Initialize and train a SVGPR model with 100 inducing points
california_trainer_svgp = SparseGPTrainer(X_train, y_train, num_inducing_points=100, num_epochs=30, batch_size=512, lr=0.1, patience=3)
california_trainer_svgp.train()

# MDN: Initialize and train an MDN model
california_trainer_mdn = MDNTrainer(X_train, y_train, num_epochs=100, lr=0.001, dense1_units=50, n_gaussians=10)
california_trainer_mdn.train()

# Set an index for plotting
index_instance = 900

# SVPGR: Visualize the SVGPR's predictions for a single instance
plt.show = lambda: plt.savefig('svgpr_solo_plot.png', dpi=300)
compare_distributions_svgpr(trainer = california_trainer_svgp, x_instance = X_test[index_instance,], y_actual = y_test[index_instance])

# clear the plot
plt.clf()

# MDN: Visualize the MDN's predictions for a single instance
plt.show = lambda: plt.savefig('mdn_solo_plot.png', dpi=300)
compare_distributions_mdn(trainer = california_trainer_mdn, x_instance = X_test[index_instance,], y_actual = y_test[index_instance])

# clear the plot
plt.clf()

# SVPGR: Visualize the SVGPR's predictions for multiple instances
plt.show = lambda: plt.savefig('svgpr_grid_plot.png', dpi=300)
plot_results_grid(trainer=california_trainer_svgp, compare_func=compare_distributions_svgpr, X_test=X_test, Y_test=y_test, indices=[900, 500], ncols=2)

# clear the plot
plt.clf()

# MDN: Visualize the MDN's predictions for multiple instances
plt.show = lambda: plt.savefig('mdn_grid_plot.png', dpi=300)
plot_results_grid(trainer=california_trainer_mdn, compare_func=compare_distributions_mdn, X_test=X_test, Y_test=y_test, indices=[900, 500], ncols=2)

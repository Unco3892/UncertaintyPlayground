## Real data

This example uses the [California Housing dataset from Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). You can train and visualize the results of the models in the following way:

```python
from uncertaintyplayground.trainers.svgp_trainer import SparseGPTrainer
from uncertaintyplayground.trainers.mdn_trainer import MDNTrainer
from uncertaintyplayground.predplot.svgp_predplot import compare_distributions_svgpr
from uncertaintyplayground.predplot.mdn_predplot import compare_distributions_mdn
from uncertaintyplayground.predplot.grid_predplot import plot_results_grid
import torch
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

# SVPGR: Visualize the SVGPR's predictions for multiple instances
plot_results_grid(trainer=california_trainer_svgp, compare_func=compare_distributions_svgpr, X_test=X_test, Y_test=y_test, indices=[900, 500], ncols=2)

# MDN: Visualize the MDN's predictions for multiple instances
plot_results_grid(trainer=california_trainer_mdn, compare_func=compare_distributions_mdn, X_test=X_test, Y_test=y_test, indices=[900, 500], ncols=2)
```

Please note that these models have not been tuned for this dataset, and are only used as an example.

## Synthetic data

It can be difficult to find truely multi-modal datsets, therefore, to check that MDN works as epected, we can capture the relationships with some simulated data:

```python
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
```


# **UncertaintyPlayground**

![CI Test Suite](https://github.com/unco3892/UncertaintyPlayground/actions/workflows/ci_cd.yml/badge.svg?branch=main)
[![Python Version](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/UncertaintyPlayground.svg)](https://pypi.org/project/UncertaintyPlayground/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Installation

*Requirements*:
- Python >= 3.8
- PyTorch == 2.0.1
- GPyTorch == 1.10
- Numpy == 1.24.0
- Seaborn == 0.12.2

Use `PyPI` to install the package:
```bash
pip install uncertaintyplayground
```

or alterntively, to use the development version, install directly from GitHub:
```bash
pip install git+https://github.com/unco3892/UncertaintyPlayground.git
```

## Usage

You can train and visualize the results of the models in the following way (this example uses the [California Housing dataset from Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)):

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

You can find another example for MDN in the `examples` folder.

<!-- ## Examples, Tutorials, and Documentation -->

## Contributors

This library is maintained by [Ilia Azizi](https://iliaazizi.com/) (University of Lausanne). Any other contributors are welcome to join! Feel free to get in touch with (contact links on my website).
<!-- Please see the [contributing guide](CONTRIBUTING.md) for more details. -->

## Citation

If you use this package in your research, please cite our work:

**UncertaintyPlayground: A Fast and Simplified Python Library for Uncertainty Estimation** , Ilia Azizi, [arXiv:2310.15281](https://arxiv.org/abs/2310.15281)

```bibtex
@misc{azizi2023uncertaintyplayground,
      title={UncertaintyPlayground: A Fast and Simplified Python Library for Uncertainty Estimation}, 
      author={Ilia Azizi},
      year={2023},
      eprint={2310.15281},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

## License

Please see the project MIT licensed [here](LICENSE).

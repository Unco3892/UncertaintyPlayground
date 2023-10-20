# **UncertaintyPlayground**

![CI Test Suite](https://github.com/unco3892/UncertaintyPlayground/actions/workflows/ci_test.yml/badge.svg?branch=main)
[![Python Version](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
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

or alterntively, clone the repo and from the root directory of the repo, run the following in your terminal:

```bash
pip install .
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

## Package structure

```bash
UncertaintyPlayground/
├── README.md
├── setup.py
├── LICENSE
├── MANIFEST.in
├── tox.ini # Local continuous integration
├── mkdocs.yml
├── docs # Documentation
│   ├── bib.md
│   ├── example.md
│   ├── gen_ref_pages.py
│   ├── index.md
│   └── README.md
├── examples # Examples of the package
│   ├── compare_models_example.py
│   ├── mdn_example.py
│   └── svgp_example.py
└── uncertaintyplayground # Main package
    ├── requirements.txt
    ├── models # Models
    │   ├── mdn_model.py
    │   └── svgp_model.py
    ├── trainers # Training the models
    │   ├── base_trainer.py
    │   ├── mdn_trainer.py
    │   └── svgp_trainer.py
    ├── predplot # Single instance prediction (inference) plot
    │   ├── grid_predplot.py
    │   ├── mdn_predplot.py
    │   └── svgp_predplot.py
    ├── utils # Utility functions
    │   ├── early_stopping.py
    │   └── generate_data.py
    └── tests # Unit tests
        ├── test_early_stopping.py
        ├── test_generate_data.py
        ├── test_mdn_model.py
        ├── test_mdn_predplot.py
        ├── test_mdn_trainer.py
        ├── test_svgp_model.py
        ├── test_svgp_predplot.py
        └── test_svgp_trainer.py
```

## Contributors

`UncertinatyPlayground` is maintained by [Ilia Azizi](https://iliaazizi.com/) (University of Lausanne). Any other contributors are welcome to join! Feel free to get in touch with (contact links on my website).
<!-- Please see the [contributing guide](CONTRIBUTING.md) for more details. -->

## License

Please see the project MIT licensed [here](LICENSE).

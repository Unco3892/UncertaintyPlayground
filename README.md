
# Installation

Requirements
- Python >= 3.8
- PyTorch >= 1.8.1
- GPyTorch >= 1.5.0
- Pyro >= 1.6.0

## Installation
From the root directory of the repo, please run the following command in your terminal:
```bash
pip install .
```

# Examples, Tutorials, and Documentation

## Testing
```python
import uncertaintyplayground as up
```

# Package structure

<!-- ![CI Test Suite](https://github.com/unco3892/UncertaintyPlaygroundV1/actions/workflows/ci_test.yml/badge.svg?branch=main) -->
[![Documentation Status](0.1)](docs/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![Python Version](https://img.shields.io/badge/python-3.8+-red.svg)](https://www.python.org/downloads/)

```bash
UncertaintyPlayground/
├── uncertaintyplayground/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── svgp.py  # Contains both regression and classification models
│   │   └── mdn.py   # Contains both regression and classification models
│   ├── utils/
│   │   ├── __init__.py
│   │   └── early_stopping.py
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── base_trainer.py
│   │   ├── svgp_trainer.py  # Contains both regression and classification trainers
│   │   └── mdn_trainer.py  # Contains both regression and classification trainers
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_models.py/
│   │       ├── __init__.py
│   │       ├── test_svgp_regression.py  # Tests for the SVGP regression model
│   │       ├── test_svgp_classification.py  # Tests for the SVGP classification model
│   │       ├── test_mdn_regression.py  # Tests for the MDN regression model
│   │       └── test_mdn_classification.py  # Tests for the MDN classification model
├── docs/
│   ├── index.rst
│   └── modules.rst
├── examples/
│   ├── svgp_example.py # Contains SVGP both regression and classification examples
│   └── mdn_example.py # Contains MDN both regression and classification examples
├── benchmarks/
│   ├── benchmark_regression.py
│   └── benchmark_classification.py
├── setup.py
├── README.md
└── .gitignore

```

Or for the docs, should do the following instead?

```bash
|-- docs/
|   |-- index.md
|   |-- installation.md
|   |-- usage.md
```

Also, how should I define the modules for the testing part? Should they be seperate? Look at the best practices for how to place your tests.

Other TODOS:
- Can also use other kernels than RBF -> Option can also be added so that it's modular for that too
- Adding multi-GPU support -> At the moment, we use the best hardware available

To use SVGP for both regression and classification, we must:
- Change the likelihood: For classification problems, you'll typically use a BernoulliLikelihood or SoftmaxLikelihood depending on whether it's binary or multiclass classification.
- Change the performance metric: Accuracy, AUC, or F1-score might be a more appropriate metric than MSE for classification.
- Change the loss function: Instead of the Mean Squared Error Loss, you might want to use Binary Cross-Entropy (for binary classification) or Cross-Entropy Loss (for multi-class classification).

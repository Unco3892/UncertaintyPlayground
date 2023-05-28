# UncertaintyPlayground: Fast *(and easy)* estimation of prediction intervals with neural networks

<!-- CI test badge will be added once the repo is made public -->
<!-- ![CI Test Suite](https://github.com/unco3892/UncertaintyPlayground/actions/workflows/ci_test.yml/badge.svg?branch=main) -->
[![Python Version](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Installation

*Requirements*:
- Python >= 3.8
- PyTorch == 2.0.1
- GPyTorch == 1.10
- Numpy == 1.24.0
- Seaborn == 0.12.2

From the root directory of the repo, run the following in your terminal:
```bash
pip install .
```

Then, you can import the module:

```python
import uncertaintyplayground as up
```

# Examples, Tutorials, and Documentation

# Package structure

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


## Further Development
Here are some ideas on how to this packaeg can be further developed:
- Can also use other kernels than RBF -> Option can also be added so that it's modular for that too
- Adding multi-GPU support -> At the moment, we use the best hardware available
- Use SVGP also for classification, we requires:
    - Change the likelihood: For classification problems, you'll typically use a BernoulliLikelihood or SoftmaxLikelihood depending on whether it's binary or multiclass classification.
    - Change the performance metric: Accuracy, AUC, or F1-score might be a more appropriate metric than MSE for classification.
    - Change the loss function: Instead of the Mean Squared Error Loss, you might want to use Binary Cross-Entropy (for binary classification) or Cross-Entropy Loss (for multi-class classification).

## Contributors

`UncertinatyPlayground` is maintained by [Ilia Azizi](https://iliaazizi.com/) (University of Lausanne). Any other contributors are welcome to join! Feel free to get in touch with (contact links on my website).
<!-- Please see the [contributing guide](CONTRIBUTING.md) for more details. -->

## License

Please see the project MIT licensed [here](LICENSE).
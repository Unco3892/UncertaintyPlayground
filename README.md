# `UncertaintyPlayground`
Fast *(and easy)* estimation of prediction intervals with neural networks

<!-- CI test badge will be added once the repo is made public -->
<!-- ![CI Test Suite](https://github.com/unco3892/UncertaintyPlayground/actions/workflows/ci_test.yml/badge.svg?branch=main) -->
[![Python Version](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Installation

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
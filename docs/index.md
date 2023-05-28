# **Uncertainty Playground: Fast estimation of prediction intervals with neural networks**

This Python library provides fast ( and *easy*) uncertainty estimation for regression tasks built on top of `PyTorch` & `GPyTorch.` Specifically, the library uses `Sparse & Variational Gaussian Process Regression` for Gaussian cases (normally distributed outcomes) and `Mixed Density Networks` for multi-modal cases. Users can estimate the prediction interval for a given instance with either model. The library was developed and maintained by [Ilia Azizi](https://iliaazizi.com/). Please note that this version is still in early development and not ready for production use. 

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


## Documentation layout

POST THE FINAL LAYOUT HERE

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.


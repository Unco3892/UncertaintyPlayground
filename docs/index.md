# **Uncertainty Playground: A Fast and Simplified Python Library for Uncertainty Estimation**

![CI Test Suite](https://github.com/unco3892/UncertaintyPlayground/actions/workflows/ci_cd.yml/badge.svg?branch=main)
[![Python Version](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/UncertaintyPlayground.svg)](https://pypi.org/project/UncertaintyPlayground/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This Python library provides fast (and *easy*) prediction intervals for regression tasks built on top of `PyTorch` & `GPyTorch`. Specifically, the library uses `Sparse & Variational Gaussian Process Regression` for Gaussian cases (normally distributed outcomes) and `Mixed Density Networks` for multi-modal distributions. Users can estimate the prediction interval for a given instance with either model. The library was developed and maintained by [Ilia Azizi](https://iliaazizi.com/). Please note that this version is still in early development and not ready for production use. 

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

Then, you can import the module:

```python
import uncertaintyplayground as up
```

## Documentation layout

Aside from this page, there are three other sections in this documentation. The most important is the `Code Reference` which relates the source code and all the arguments for the functions.  The `Usage` section contains a couple of examples of using this package with real and simulated data. Finally, the `Bibliography` section contains the list of papers that are used in this package.

## Contributors

This library is maintained by [Ilia Azizi](https://iliaazizi.com/) (University of Lausanne). Any other contributors are welcome to join! Feel free to get in touch with (contact links on my website).
<!-- Please see the [contributing guide](CONTRIBUTING.md) for more details. -->

## Citation

If you use this package in your research, please cite our work:

**UncertaintyPlayground: A Fast and Simplified Python Library for Uncertainty Estimation**, Ilia Azizi, [arXiv:2310.15281](https://arxiv.org/abs/2310.15281)

``` bibtex
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

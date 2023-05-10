Package structure

FIX THIS AT THE END

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

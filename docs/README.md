The docs were built with `mkdocs` and `mcdocstrings`. To navigate the HTML, please run the following commands (one by one):

```bash
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ python -m pip install mkdocs
(venv) $ python -m pip install "mkdocstrings[python]"
(venv) $ python -m pip install mkdocs-material
(venv) $ pip install mkdocs-gen-files
(venv) $ pip install mkdocs-literate-nav
(venv) $ pip install mkdocs-section-index
```

Or alternatively, this one liner:

```bash
python3 -m venv venv && source venv/bin/activate && python -m pip install mkdocs "mkdocstrings[python]" mkdocs-material mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index
```

Then at the root of the project, feel free to run the following commands to read the documentations:

* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

More info can be found here https://mkdocstrings.github.io/recipes/
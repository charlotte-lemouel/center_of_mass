# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Center of Mass'
copyright = '2024, Charlotte LE MOUEL'
author = 'Charlotte LE MOUEL'

# The full version, including alpha/beta/rc tags
release = 'v1.0'


# -- General configuration ---------------------------------------------------

# this is a trick to make sphinx find the modules in the parent directory
import os
import sys
# sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("..\\src"))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['nbsphinx', # for jupyter notebooks
'myst_parser', # to read markdown (.md) files
#'myst_nb', # for jupyter notebooks
'numpydoc', # support for the Numpy docstring format: when you have this you don't need sphinx.ext.autodoc
#"sphinx.ext.autodoc"#  Include documentation from docstrings
'sphinx.ext.imgmath' # renders math formulas as images
]

myst_enable_extensions = ["amsmath"] # for Latex equations

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import inspect

sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'toad'
copyright = '2020, ESC Team'
author = 'ESC Team'


import toad
version = toad.VERSION
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------
import recommonmark
import sphinx_readable_theme
from recommonmark.transform import AutoStructify

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'recommonmark',
    'sphinx_readable_theme',
]



autodoc_member_order = 'bysource'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    'toad/commands',
    '_build',
    '**.ipynb_checkpoints',
]

master_doc = 'index'


def linkcode_resolve(domain, info):
    """linkcode extension config function
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        # inspect.unwrap() was added in Python version 3.4
        if sys.version_info >= (3, 5):
            fn = inspect.getsourcefile(inspect.unwrap(obj))
        else:
            fn = inspect.getsourcefile(obj)
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = "#L{:d}-L{:d}".format(lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    fn = os.path.relpath(fn, start = os.path.dirname(toad.__file__))

    return "http://github.com/amphibian-dev/toad/blob/master/toad/{}{}".format(
        fn, linespec
    )


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
html_theme = 'readable'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']



def setup(app):
    app.add_config_value(
        'recommonmark_config',
        {
            'enable_eval_rst': True,
            'enable_auto_toc_tree': True,
            'auto_toc_tree_section': 'Contents',
        },
        True,
    )

    app.add_transform(AutoStructify)

# Build Doxygen docs
import os, subprocess
onRtd = os.environ.get('READTHEDOCS') == 'True'
if not os.path.exists('_build'):
    os.makedirs('_build')
if not os.path.exists('_build/doxygen'):
    os.makedirs('_build/doxygen')
subprocess.call('cd _build/doxygen; doxygen ../../Doxyfile', shell=True)
if not onRtd:
    if not os.path.exists('DoxygenOutput'):
        os.makedirs('DoxygenOutput')
    subprocess.call('cp -r _build/doxygen/html DoxygenOutput/', shell=True)

# -- Project information -----------------------------------------------------

project = 'TFC'
copyright = '2020, Carl Leake, Hunter Johnston'
author = 'Carl Leake, Hunter Johnston'

# -- General configuration ---------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..','src')))

# Minimal version of sphinx needed
needs_sphinx = '2.1'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['breathe',
              'exhale',
              'nbsphinx',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx_copybutton']

# Breathe Configuration
breathe_default_project = "TFC"
breathe_projects = {"TFC":"_build/doxygen/xml"}

# Exhale Configuration
exhale_args = {
        "containmentFolder":"./Exhale",
        "rootFileName":"exhale_root.rst",
        "rootFileTitle":"C++ Documentation",
        "doxygenStripFromPath":"..",
        "createTreeView":True
        }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
if onRtd:
    html_static_path = ['_build/doxygen/html']
else:
    html_static_path = ['DoxygenOutput/html']

# Choose Pygments style
pygments_style = None

# Mock import for mayavi so it does not have to be installed as a dependency in ReadTheDocs
autodoc_mock_imports = ['mayavi']

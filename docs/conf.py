# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Moirae'
copyright = '2024'
author = 'Victor Venturi, Logan Ward, Argonne National Laboratory'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- Options for NBSphinx -----------------------------------------------------

extensions.append('nbsphinx')
nbsphinx_execute = 'never'

# -- API Documentation --------------------------------------------------------

extensions.extend([
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx-pydantic',
    'sphinxcontrib.autodoc_pydantic'
])

autodoc_mock_imports = ["thevenin"]

autodoc_pydantic_model_show_json = False
autodoc_pydantic_settings_show_json = False

autoclass_content = 'both'

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'thevenin': ('https://thevenin.readthedocs.io/stable/', None),
    'batdata': ('https://rovi-org.github.io/battery-data-toolkit/', None),
}

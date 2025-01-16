# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CCN software workshop, January 2025'
copyright = '2025, Edoardo Balzani, Billy Broderick, Sarah Jo Venditto, Guillaume Viejo'
author = 'Edoardo Balzani, Billy Broderick, Sarah Jo Venditto, Guillaume Viejo'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_togglebutton',
    'sphinx_design',
    'sphinx.ext.intersphinx'
]

# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {'nemos': ("https://nemos.readthedocs.io/en/latest/", None)}

templates_path = []
exclude_patterns = []

nitpicky = True
# raise an error if exec error in notebooks
nb_execution_raise_on_error = True

sphinxemoji_style = 'twemoji'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# max time (in secs) per notebook cell. here, we disable this
nb_execution_timeout = -1
# we have two versions of each notebook, one with explanatory text and one without
# (which ends in `-stripped.md`). we don't need to run both of them
nb_execution_excludepatterns = ['*stripped*']
nb_execution_raise_on_error = True
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_favicon = '_static/ccn_small.png'
html_sourcelink_suffix = ""
myst_enable_extensions = ["colon_fence", "dollarmath", "attrs_inline"]
html_theme_options = {
    "home_page_in_toc": True,
    "github_url": "https://github.com/flatironinstitute/ccn-software-jan-2025",
    "repository_url": "https://github.com/flatironinstitute/ccn-software-jan-2025",
    "logo": {
        "alt_text": "Home",
        "image_light": "_static/01-FI-primary-logo-color.png",
        "image_dark": "_static/03-FI-primary-logo-white.png",
    },
    "use_download_button": True,
    "use_repository_button": True,
    "icon_links": [
        {
            "name": "Workshops home",
            "url": "https://flatironinstitute.github.io/neurorse-workshops/",
            "type": "fontawesome",
            "icon": "fa-solid fa-house",
        },
        {
            "name": "Binder",
            "url": "https://binder.flatironinstitute.org/v2/user/wbroderick/jan2025?labpath=notebooks/",
            "type": "url",
            "icon": "https://mybinder.org/badge_logo.svg",
        },
    ],
}
nb_execution_excludepatterns = ['*model_selection*', '*-users*', '*-presenters*']
nb_execution_mode = "cache"

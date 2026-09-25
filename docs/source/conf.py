"""Sphinx configuration for the Fermi documentation."""

from datetime import date
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import fermi


project = "fermi"
author = "CREF team"
copyright = f"2025-{date.today().year}, {author}"
version = fermi.__version__
release = fermi.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

master_doc = "index"
templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

html_theme = "alabaster"
html_title = f"Fermi {release} documentation"
html_static_path = []
html_theme_options = {
    "description": "Fitness, Relatedness, and other Economic Complexity metrics",
    "github_user": "EFC-data",
    "github_repo": "fermi",
    "github_button": True,
}

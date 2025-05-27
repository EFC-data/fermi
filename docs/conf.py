import os
import sys

# punta al folder che contiene il package “fermi”
sys.path.insert(0, os.path.abspath('..'))


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]
html_theme = 'sphinx_rtd_theme'
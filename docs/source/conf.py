# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'Target Finder Toolkit'
copyright = '2025, Anonymous'
author = 'Anonymous'
release = '0.1.3'

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'target_finder_toolkit')))

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',     
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',    
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = 'description'   
autoclass_content = "both"

autodoc_default_options = {
    "members": True,          
    "undoc-members": False,  
    "private-members": False, 
    "special-members": False,
    "inherited-members": False,
    "show-inheritance": True,
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'   
html_static_path = ['_static']

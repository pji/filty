"""
__init__
~~~~~~~~

Initialization for the filty module.
"""
from filty.filty import *
from filty import filty
from filty.utility import get_prefixed_functions


# Create a dictionary to allow easier discovery and validation of
# the filters available in the module.
filters = get_prefixed_functions('filter_', filty)

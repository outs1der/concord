import pkg_resources

CONCORD_PATH = pkg_resources.resource_filename('concord','data')

__author__ = 'Duncan K. Galloway'
__email__ = 'Duncan.Galloway@monash.edu'
__version__ = '1.1.1'

# Ensure objects are available at the package level

# from concord import burstclass
from .utils import *
from .burstclass import *


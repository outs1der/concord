import pkg_resources

CONCORD_PATH = pkg_resources.resource_filename('concord','data')

# Ensure objects are available at the package level

# from concord import burstclass
from .utils import *
from .burstclass import *


import importlib
from importlib.metadata import version as get_version

pkg_name = importlib.import_module(__name__).__package__.split('.')[0]
pkg_version = get_version("deepx-pack")

__version__ = pkg_version


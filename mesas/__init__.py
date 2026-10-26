"""MESAS - Multiscale Estimation of StorAge Selection."""

from importlib.metadata import PackageNotFoundError, version

from mesas.sas.model import Model, ModelOptions, ModelResult, SoluteSpec

try:
    __version__ = version("mesas")
except PackageNotFoundError:
    # Running from an uninstalled source tree
    __version__ = "2.0.0+dev"

__all__ = ["Model", "ModelOptions", "ModelResult", "SoluteSpec", "__version__"]

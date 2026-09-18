"""Compressible flow of a mixture with diffusion, and its boundaries."""

from .boundaries import BaseNSBC
from .simulator import NavierStokesSimulator

__all__ = ["BaseNSBC", "NavierStokesSimulator"]

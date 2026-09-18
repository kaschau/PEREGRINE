"""Compressible flow of a mixture with diffusion, and its boundaries."""

from .boundaries import BaseNSBC
from .simulation import NavierStokesSimulation

__all__ = ["BaseNSBC", "NavierStokesSimulation"]

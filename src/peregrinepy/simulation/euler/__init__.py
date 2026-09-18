"""Compressible flow of a mixture without diffusion, and its boundaries."""

from .boundaries import BaseEulerBC
from .simulation import EulerSimulation

__all__ = ["BaseEulerBC", "EulerSimulation"]

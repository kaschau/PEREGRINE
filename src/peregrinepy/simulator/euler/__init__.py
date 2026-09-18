"""Compressible flow of a mixture without diffusion, and its boundaries."""

from .boundaries import BaseEulerBC
from .simulator import EulerSimulator

__all__ = ["BaseEulerBC", "EulerSimulator"]

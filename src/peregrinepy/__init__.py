from . import compute_ as compute

if compute.KokkosLocation in ['OpenMP','CudaUVM','Serial','Default']:
    import numpy as np
else:
    raise ValueError(f'Unknown KokkosLocation {compute.KokkosLocation}')

from .multiblock import multiblock
from .block import block
from .construct_mb import construct_mb

from . import files
from . import grid
from . import readers
from . import writers
from . import mpicomm

from . import bcs
from .consistify import consistify

from . import rk1,rk4

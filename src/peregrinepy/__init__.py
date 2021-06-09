from . import compute_ as compute

if compute.KokkosLocation in ['OpenMP','CudaUVM','Default']:
    import numpy as np
else:
    raise ValueError(f'Unknown KokkosLocation {compute.KokkosLocation}')

from .multiblock import multiblock
from .block import block
from . import files
from . import initialize
from . import readers
from . import writers
from . import mpicomm

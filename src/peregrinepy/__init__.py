from . import _compute as compute

if compute.KokkosLocation in ['OpenMP','CudaUVM','Default']:
    import numpy as np
else:
    raise ValueError(f'Unknown KokkosLocation {compute.KokkosLocation}')

from .block import block
from . import initialize
from . import readers

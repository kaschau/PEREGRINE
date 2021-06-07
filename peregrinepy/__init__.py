import sys
from kokkos import HostSpace, CudaUVMSpace, CudaSpace

#We make the KokkosLocation variable a module global
#as well as setting the python side array module to
#either numpy or cupy based on the executaion space.
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent / "../Lib/"))
from Peregrine import KokkosLocation

if KokkosLocation in ['Serial','OpenMP','Default']:
    import numpy as np
    globals()['device_array'] = np
    globals()['space'] = HostSpace
if KokkosLocation == 'CudaUVM':
    import numpy as np
    globals()['device_array'] = np
    globals()['space'] = CudaUVMSpace
elif KokkosLocation == 'Cuda':
    import cupy as cp
    globals()['device_array'] = cp
    globals()['space'] = CudaSpace
else:
    raise ValueError(f'Unknown KokkosLocation {KokkosLocation}.')

#Now the rest of the stuff
from .block import block
from .initialize_arrays import initialize_arrays

from . import readers
from . import compute

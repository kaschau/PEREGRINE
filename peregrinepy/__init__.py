import sys
import kokkos

#We make the KokkosLocation variable a module global
#as well as setting the python side array module to
#either numpy or cupy based on the executaion space.
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent / "../Lib/"))
import Peregrine
if Peregrine.KokkosLocation in ['Serial','OpenMP','Default']:
    import numpy as np
    globals()['device_array'] = np
    globals()['space'] = kokkos.HostSpace
    #def array():
    #    return np
else:
    import cupy as cp
    globals()['device_array'] = cp
    if Peregrine.KokkosLocation == 'CudaUVM':
        globals()['space'] = kokkos.CudaUVMSpace
    elif Peregrine.KokkosLocation == 'Cuda':
        globals()['space'] = kokkos.CudaSpace

#Now the rest of the stuff
from .block import block
from .initialize_arrays import initialize_arrays

from . import readers
from . import compute

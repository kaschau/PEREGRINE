import sys

# We need the KokkosLocation variable to be available on import
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent / "../Lib/"))
import Peregrine
def KokkosLocation():
    return Peregrine.KokkosLocation

#Now the rest of the stuff
from .block import block
from . import readers
from .initialize_arrays import initialize_arrays
from .add3 import add3

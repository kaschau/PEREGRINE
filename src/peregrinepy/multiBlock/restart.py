from .arrays import CellCenterArray
from .grid import grid
from .restartBlock import restartBlock


class restart(grid):
    """A grid with a state on it: the primitive vector of every block at
    one time, as wide as its primitive variables -- p, u, v, w, T and
    whatever a physics adds -- which are what a result file writes and a
    reader matches."""

    def _newBlock(self, nblki):
        return restartBlock(nblki, self)

    def __init__(self, primVars):
        super().__init__()
        self.primVars = list(primVars)
        self.ne = len(self.primVars)
        # what a result of this multiBlock writes: the vector, nothing derived
        self.exportVars = self.primVars
        # the step count and the time the state is at
        self.nrt = 0
        self.tme = 0.0
        # the primitive vector, what a result file holds of the state
        self.declArray("prims", CellCenterArray, components=self.ne)

    def exportData(self, blk, names):
        """Gives the named primitive variables of a block as host arrays
        over every cell, halos included."""
        prims = blk.prims.get()
        return {name: prims[..., self.primVars.index(name)] for name in names}

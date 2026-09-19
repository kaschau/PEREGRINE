from ..backend.abi import pgDims
from .restartBlock import restartBlock
from .solverFace import solverFace


class solverBlock(restartBlock):
    """A block of a solver: it puts its arrays in the tables the launches
    read."""

    def __init__(self, nblki, mb):
        # the solver this block belongs to
        self.mb = mb
        super().__init__(nblki, mb)

    def tableValue(self, name):
        """Gives what this block puts in a table under a name: an array, or
        its dims -- None for one it does not hold."""
        return getattr(self, name, None)

    @property
    def dims(self):
        return pgDims.of(self)

    def _newFace(self, nface):
        return solverFace(nface, self)

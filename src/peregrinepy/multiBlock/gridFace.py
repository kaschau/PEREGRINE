from ..abi import HostStorageMixin
from .topologyFace import topologyFace


class gridFace(topologyFace, HostStorageMixin):
    """
    gridFace object holds all the information that a grid
    would need to know about a face.
    """

    def __init__(self, nface):
        super().__init__(nface)

        # every array is an attribute named for it, shaped once the block is
        # sized; what a face declares is all it may hold
        self.declared = {}

        self.declare("periodicRotMatrix", kind="rotation")

    ###########################################################################
    # The arrays a face has, and how big they are
    ###########################################################################
    @property
    def shapes(self):
        """What each kind of array is shaped, for this face's block."""
        return {"rotation": (3, 3)}

    def allocate(self, *names, **values):
        """Give these arrays their memory now that their shapes are known,
        and any given by name their values. A face allocates in groups rather
        than all at once, since an interior face never holds boundary values
        and a boundary face never holds halo buffers."""
        for name in names:
            setattr(self, name, self._new(self.shapeOf(name)))
        for name, array in values.items():
            setattr(self, name, self._new(self.shapeOf(name)))
            self.store(name, array)

    ###########################################################################
    # How a halo arriving through this face is moved onto it
    ###########################################################################
    @topologyFace.periodicRotation.setter
    def periodicRotation(self, rotation):
        topologyFace.periodicRotation.fset(self, rotation)
        # the compute side turns the vectors in the halo with the same matrix,
        # so it needs one of its own to read
        if rotation is None:
            return
        self.allocate(periodicRotMatrix=rotation)

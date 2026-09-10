import numpy as np

from .topologyFace import topologyFace


class gridFace(topologyFace):
    """
    gridFace object holds all the information that a grid
    would need to know about a face.
    """

    faceType = "grid"

    def __init__(self, nface):
        super().__init__(nface)

        #########################################################
        # Data arrays
        #########################################################
        # Python side data
        self.array = {}
        # Kokkos mirrors (only used for solverFaces)
        self.mirror = {}
        # what each array's shape will be, once the block is sized
        self.declared = {}

        self.declare("periodicRotMatrix", kind="rotation")

    ###########################################################################
    # The arrays a face has, and how big they are
    ###########################################################################
    def declare(self, *names, kind):
        """Say an array exists and what shape it will take, before the block
        this face bounds knows its extents. Nothing else may be put in array."""
        for name in names:
            self.declared[name] = kind
            self.array[name] = None
            self.mirror[name] = None

    @property
    def shapes(self):
        """What each kind of array is shaped, for this face's block."""
        return {"rotation": (3, 3)}

    def shapeOf(self, name):
        return self.shapes[self.declared[name]]

    def allocate(self, *names):
        """Give these arrays their memory now that their shapes are known. A
        face allocates in groups rather than all at once, since an interior
        face never holds boundary values and a boundary face never holds
        halo buffers."""
        for name in names:
            self.array[name] = np.zeros(self.shapeOf(name))

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
        self.allocate("periodicRotMatrix")
        self.array["periodicRotMatrix"][:] = rotation

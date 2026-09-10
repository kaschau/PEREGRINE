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

        self.declare("periodicRotMatrixUp", "periodicRotMatrixDown", kind="rotation")

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
    # Periodic rotation
    ###########################################################################
    @topologyFace.periodicAxis.setter
    def periodicAxis(self, axis):
        topologyFace.periodicAxis.fset(self, axis)

        # only a rotational periodic turns anything, and it cannot know how
        # far around until it has been given its span
        if not self.bcType.startswith("periodicRot"):
            return
        if self.periodicSpan is None:
            raise AttributeError("Must set periodicSpan before setting periodicAxis")

        self.allocate("periodicRotMatrixUp", "periodicRotMatrixDown")
        up = self._rotationMatrix(self.periodicAxis, self.periodicSpan * np.pi / 180.0)
        self.array["periodicRotMatrixUp"][:] = up
        # a rotation is orthogonal, so turning back the way we came is its
        # transpose
        self.array["periodicRotMatrixDown"][:] = up.T

    @staticmethod
    def _rotationMatrix(u, theta):
        """Turning about the unit vector :u: by :theta:.
        See http://paulbourke.net/geometry/rotate/"""
        ct, st = np.cos(theta), np.sin(theta)
        cross = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
        return ct * np.eye(3) + st * cross + (1.0 - ct) * np.outer(u, u)

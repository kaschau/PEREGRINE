import numpy as np

from .topologyFace import topologyFace


class gridFace(topologyFace):
    """A block face with coordinates behind it: the planes of a block array
    either side of it, as the halo is generated and moved, and the array
    only a block face has -- the rotation a periodic halo is turned by,
    made when set. Nothing else is put in an array attribute."""

    def __init__(self, nface, backend, ng):
        super().__init__(nface)
        # the backend its arrays are made on, and the halo depth of the block
        self.backend = backend
        self.ng = ng
        self.rotation = None

    def allocate(self, name, shape):
        """Makes the named array, zeroed, of this shape, on this face's
        backend."""
        setattr(self, name, self.backend.allocate(shape, name=name))

    ###########################################################################
    # The planes of a block array either side of this face, as the kernels
    # walk them: the layer first, layer 0 nearest the face
    ###########################################################################
    def halo(self, array):
        """Gives the halo planes, outward from the face."""
        ng = self.ng
        planes = np.moveaxis(array, self.myAxis, 0)
        return planes[ng - 1 :: -1] if self.amILow else planes[-ng:]

    def interior(self, array, skip=0):
        """Gives the interior planes, inward from the face; :skip: leaves
        out the first, for a node array whose face plane both sides hold."""
        ng = self.ng
        planes = np.moveaxis(array, self.myAxis, 0)
        if self.amILow:
            return planes[ng + skip : 2 * ng + skip]
        return planes[-(ng + 1 + skip) : -(2 * ng + 1 + skip) : -1]

    def blockFacePlane(self, array):
        """Gives the plane of a cell-face array lying on this block face,
        over the block face proper."""
        ng = self.ng
        planes = np.moveaxis(array, self.myAxis, 0)
        return planes[ng if self.amILow else -(ng + 1)][ng:-ng, ng:-ng]

    ###########################################################################
    # How a halo arriving through this face is moved onto it
    ###########################################################################
    def setPeriodic(self, rotation=None, translation=None):
        """Sets the transform, and puts the rotation where the kernels run:
        they turn the vectors in the halo with the same matrix."""
        super().setPeriodic(rotation, translation)
        self.allocate("rotation", (3, 3))
        self.rotation.set(self.periodicRotation)

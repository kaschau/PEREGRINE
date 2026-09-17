from .topologyFace import topologyFace


class gridFace(topologyFace):
    """A block face with arrays of its own: only what a block face does --
    the rotation a periodic halo is turned by here; a solver face adds its
    bc values and its trade buffers -- each stored as face.<name>, None
    until made. Nothing else is put in an array attribute."""

    def __init__(self, nface, backend):
        super().__init__(nface)
        # the backend its arrays are made on
        self.backend = backend
        # the rotation a periodic halo is turned by, where the kernels run
        self.rotation = None

    def allocate(self, name, shape):
        """Makes the named array, zeroed, of this shape, on this face's
        backend."""
        setattr(self, name, self.backend.allocate(shape, name=name))

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
        self.allocate("rotation", (3, 3))
        self.rotation.set(rotation)

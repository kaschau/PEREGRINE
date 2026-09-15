from .topologyFace import topologyFace


class gridFace(topologyFace):
    """A face with arrays of its own: a short list of kinds only a face has
    (the rotation here; a solver face adds its bc values and trade buffers),
    each stored as face.<name> and made by name when asked for. Nothing else
    is put in an array attribute."""

    def __init__(self, nface, backend):
        super().__init__(nface)
        # the backend its arrays are made on, and what it may hold: every
        # array is an attribute named for it, made when asked for
        self.backend = backend
        self.declared = {}
        self.declare("periodicRotMatrix", kind="rotation")

    ###########################################################################
    # The arrays a face has, and how big they are
    ###########################################################################
    def declare(self, *names, kind, components=()):
        """Arrays this face may hold, by kind; each None until made."""
        if isinstance(components, int):
            components = (components,)
        for name in names:
            self.declared[name] = (kind, components)
            setattr(self, name, None)

    @property
    def shapes(self):
        """What each kind of array is shaped, for this face's block."""
        return {"rotation": (3, 3)}

    def shapeOf(self, name):
        kind, components = self.declared[name]
        return self.shapes[kind] + components

    def allocate(self, *names):
        """The named arrays, zeroed, on this face's backend; one that already
        has its shape keeps what is in it."""
        for name in names:
            shape = self.shapeOf(name)
            current = getattr(self, name)
            if current is not None and current.shape == shape:
                continue
            kind, components = self.declared[name]
            array = self.backend.allocate(
                shape, name=name, kind=kind, components=components
            )
            setattr(self, name, array)

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
        self.periodicRotMatrix.set(rotation)

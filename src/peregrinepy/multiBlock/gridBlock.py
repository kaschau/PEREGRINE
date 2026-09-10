import numpy as np
from .topologyBlock import topologyBlock
from ..misc import frozenDict
from ..grid import metrics
from ..grid import generateHalo


class gridBlock(topologyBlock):
    """
    gridBlock object holds all the information that a grid
    would need to know about a block.
    """

    blockType = "grid"

    def __init__(self, nblki, ng=0):
        self.ng = ng

        super().__init__(nblki)

        self.ni = 0
        self.nj = 0
        self.nk = 0

        #########################################################
        # Data arrays
        #########################################################
        # Python side data
        self.array = frozenDict()
        # Kokkos mirrors (only used for solverBlocks)
        self.mirror = frozenDict()
        # Coordinate arrays
        for d in ["x", "y", "z"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # Grid metrics
        # Cell centers
        for d in ["xc", "yc", "zc", "J", "dI", "dJ", "dK"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # Cell center metrics
        for d in [
            "dEdx",
            "dEdy",
            "dEdz",
            "dNdx",
            "dNdy",
            "dNdz",
            "dCdx",
            "dCdy",
            "dCdz",
        ]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # i face area vectors
        for d in ["ixc", "iyc", "izc", "isx", "isy", "isz", "iS", "inx", "iny", "inz"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # j face area vectors
        for d in ["jxc", "jyc", "jzc", "jsx", "jsy", "jsz", "jS", "jnx", "jny", "jnz"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None
        # k face area vectors
        for d in ["kxc", "kyc", "kzc", "ksx", "ksy", "ksz", "kS", "knx", "kny", "knz"]:
            self.array[f"{d}"] = None
            self.mirror[f"{d}"] = None

        if self.blockType == "grid":
            self.array._freeze()

    def initRestartArrays(self):
        """No restart arrays on a grid block."""

    @property
    def interior(self):
        """The slice of this block's arrays that is not halo."""
        return np.s_[:, :, :]

    @property
    def baseNodeSlab(self):
        """Where this block's nodes sit in the base block it is a piece of,
        which is how the grid and result files are indexed. baseSlice counts
        nodes inclusively."""
        if self.baseSlice is None:
            return np.s_[:, :, :]
        i0, i1, j0, j1, k0, k1 = self.baseSlice
        return np.s_[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]

    @property
    def baseCellSlab(self):
        """The same, for this block's cells. A block of ni nodes spans ni-1
        cells, so nodes i0..i1 are cells i0..i1-1."""
        if self.baseSlice is None:
            return np.s_[:, :, :]
        i0, i1, j0, j1, k0, k1 = self.baseSlice
        return np.s_[k0:k1, j0:j1, i0:i1]

    def updateDeviceView(self, vars):
        """No device to push to."""

    def updateHostView(self, vars):
        """No device to pull from."""

    def initGridArrays(self):
        """
        Create zeroed numpy arrays of correct size.
        """
        ng = self.ng

        # Primary grid coordinates
        shape = [self.ni + 2 * ng, self.nj + 2 * ng, self.nk + 2 * ng]
        for name in ["x", "y", "z"]:
            self.array[name] = np.zeros((shape))

        # Cell center locations, volumes, diffusive metrics
        shape = [self.ni + 2 * ng - 1, self.nj + 2 * ng - 1, self.nk + 2 * ng - 1]
        for name in [
            "xc",
            "yc",
            "zc",
            "J",
            "dI",
            "dJ",
            "dK",
            "dEdx",
            "dEdy",
            "dEdz",
            "dNdx",
            "dNdy",
            "dNdz",
            "dCdx",
            "dCdy",
            "dCdz",
        ]:
            self.array[name] = np.zeros((shape))

        # i face normal, area vectors
        shape = [self.ni + 2 * ng, self.nj + 2 * ng - 1, self.nk + 2 * ng - 1]
        for name in [
            "ixc",
            "iyc",
            "izc",
            "isx",
            "isy",
            "isz",
            "iS",
            "inx",
            "iny",
            "inz",
        ]:
            self.array[name] = np.zeros((shape))

        # j face normal, area vectors
        shape = [self.ni + 2 * ng - 1, self.nj + 2 * ng, self.nk + 2 * ng - 1]
        for name in [
            "jxc",
            "jyc",
            "jzc",
            "jsx",
            "jsy",
            "jsz",
            "jS",
            "jnx",
            "jny",
            "jnz",
        ]:
            self.array[name] = np.zeros((shape))

        # k face normal, area vectors
        shape = [self.ni + 2 * ng - 1, self.nj + 2 * ng - 1, self.nk + 2 * ng]
        for name in [
            "kxc",
            "kyc",
            "kzc",
            "ksx",
            "ksy",
            "ksz",
            "kS",
            "knx",
            "kny",
            "knz",
        ]:
            self.array[name] = np.zeros((shape))

    def computeMetrics(self, xcOnly=False):
        metrics(self, xcOnly)

    def generateHalo(self):
        generateHalo(self)

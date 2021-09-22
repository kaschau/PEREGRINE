from numpy import s_
from ..bcs.walls import *
from ..bcs.inlets import *
from ..bcs.exits import *
from ..misc import FrozenDict


class face:
    def __init__(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        self.nface = nface
        self.connectivity = FrozenDict(
            {
                "bcfam": None,
                "bctype": "adiabatic_slip_wall",
                "neighbor": None,
                "orientation": None,
            }
        )

        self.connectivity._freeze()

        # Face slices
        if nface == 1:
            self.s0_ = s_[0, :, :]
            self.s1_ = s_[1, :, :]
            self.s2_ = s_[2, :, :]
        if nface == 2:
            self.s0_ = s_[-1, :, :]
            self.s1_ = s_[-2, :, :]
            self.s2_ = s_[-3, :, :]
        if nface == 3:
            self.s0_ = s_[:, 0, :]
            self.s1_ = s_[:, 1, :]
            self.s2_ = s_[:, 2, :]
        if nface == 4:
            self.s0_ = s_[:, -1, :]
            self.s1_ = s_[:, -2, :]
            self.s2_ = s_[:, -3, :]
        if nface == 5:
            self.s0_ = s_[:, :, 0]
            self.s1_ = s_[:, :, 1]
            self.s2_ = s_[:, :, 2]
        if nface == 6:
            self.s0_ = s_[:, :, -1]
            self.s1_ = s_[:, :, -2]
            self.s2_ = s_[:, :, -3]

        # Boundary condition values
        self.bcvals = FrozenDict({})

        # MPI variables - only set for solver blocks, but we will store them
        # all the time for now
        self.comm_rank = None
        self.neighbor_face = None
        self.neighbor_orientation = None

        self.orient = None
        self.slice_s3 = None
        self.slice_r3 = None
        self.slice_s4 = None
        self.slice_r4 = None

        self.sendbuffer3 = None
        self.recvbuffer3 = None
        self.sendbuffer4 = None
        self.recvbuffer4 = None

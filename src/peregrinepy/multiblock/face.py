from numpy import s_
from ..misc import frozenDict


class connectivityDict(frozenDict):
    def __setitem__(self, key, value):
        if key == "bcType":
            if value not in [
                "s1",
                "b0",
                "b1",
                "constant_velocity_subsonic_inlet",
                "constant_pressure_subsonic_inlet",
                "adiabatic_noslip_wall",
                "adiabatic_slip_wall",
                "adiabatic_moving_wall",
                "isoT_moving_wall",
            ]:
                raise KeyError(f"{value} is not a valid input for bcType.")
        elif key == "neighbor":
            if type(value) not in [type(None), int]:
                raise KeyError(f"{value} is not a valid input for neighbor.")
        elif key == "orientation":
            if type(value) not in [type(None), str]:
                raise KeyError(f"{value} is not a valid input for orientation.")
            if value is str and len(value) != 3:
                raise KeyError(f"{value} is not a valid input for orientation.")

        super().__setitem__(key, value)


class face:

    __slots__ = (
        "nface",
        "connectivity",
        "s0_",
        "s1_",
        "s2_",
        "bcVals",
        "commRank",
        "neighborFace",
        "neighborOrientation",
        "orient",
        "sliceS3",
        "sliceS4",
        "sliceR3",
        "sliceR4",
        "sendBuffer3",
        "sendBuffer4",
        "recvBuffer3",
        "recvBuffer4",
    )

    def __init__(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        self.nface = nface
        self.connectivity = connectivityDict(
            {
                "bcFam": None,
                "bcType": "adiabatic_slip_wall",
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
        self.bcVals = frozenDict({})

        # MPI variables - only set for solver blocks, but we will store them
        # all the time for now
        self.commRank = None
        self.neighborFace = None
        self.neighborOrientation = None

        self.orient = None
        self.sliceS3 = None
        self.sliceR3 = None
        self.sliceS4 = None
        self.sliceR4 = None

        self.sendBuffer3 = None
        self.recvBuffer3 = None
        self.sendBuffer4 = None
        self.recvBuffer4 = None

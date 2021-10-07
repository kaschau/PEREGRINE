from ..misc import frozenDict


class connectivityDict(frozenDict):
    def __setitem__(self, key, value):
        if key == "bcType":
            if value not in [
                "s1",
                "b0",
                "b1",
                "constantVelocitySubsonicInlet",
                "constantPressureSubsonicExit",
                "adiabaticNoSlipWall",
                "adiabaticSlipWall",
                "adiabaticMovingWall",
                "isoTMovingWall",
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


class topologyFace:

    __slots__ = "nface", "ng", "connectivity"

    def __init__(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        self.nface = nface
        self.connectivity = connectivityDict(
            {
                "bcFam": None,
                "bcType": "adiabaticSlipWall",
                "neighbor": None,
                "orientation": None,
            }
        )

        self.connectivity._freeze()

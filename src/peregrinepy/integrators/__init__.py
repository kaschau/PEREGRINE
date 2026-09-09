from .dualTime import dualTime
from .explicit import maccormack, rk1, rk2, rk3, rk34, rk4
from .strang import Strang

# the schemes strang can split with, and can be run on their own
_schemes = {i.integratorName: i for i in (rk1, rk2, rk3, rk34, rk4, maccormack)}
_others = {i.integratorName: i for i in (dualTime,)}

__all__ = ["getIntegrator"]


def getIntegrator(ti):
    """Named in the config. "strang" splits with rk3; "strang-rk4" or any
    other scheme name after the dash splits with that one instead."""
    split, _, scheme = ti.partition("-")
    if split == "strang":
        try:
            paired = _schemes[scheme or "rk3"]
        except KeyError:
            raise ValueError(f"Strang cannot split with {scheme}.")
        return type(ti, (Strang, paired), {"integratorName": ti})

    try:
        return {**_schemes, **_others}[ti]
    except KeyError:
        raise ValueError(f"What time integrator? {ti}")

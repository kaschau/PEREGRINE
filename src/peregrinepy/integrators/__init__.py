from .rk1 import rk1
from .rk2 import rk2
from .maccormack import maccormack
from .rk3 import rk3
from .rk34 import rk34
from .rk4 import rk4
from .strang import strang
from .dualTime import dualTime


def getIntegrator(ti):
    if ti == "rk1":
        return rk1
    elif ti == "rk2":
        return rk2
    elif ti == "maccormack":
        return maccormack
    elif ti == "rk3":
        return rk3
    elif ti == "rk34":
        return rk34
    elif ti == "rk4":
        return rk4
    elif ti == "strang":
        return strang
    elif ti == "dualTime":
        return dualTime
    else:
        raise ValueError("What time integrator?")

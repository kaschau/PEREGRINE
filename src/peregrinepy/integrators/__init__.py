from .rk1 import rk1
from .rk3 import rk3
from .rk4 import rk4
from .simpler import simpler


def get_integrator(ti):

    if ti == "rk1":
        return rk1
    elif ti == "rk3":
        return rk3
    elif ti == "rk4":
        return rk4
    elif ti == "simpler":
        return simpler
    else:
        raise ValueError("What time integrator?")

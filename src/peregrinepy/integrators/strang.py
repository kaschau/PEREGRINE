from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify
from scipy.integrate import ode
from itertools import product
from ..compute.utils import AEQB
from ..compute.timeIntegration import rk3s1, rk3s2, rk3s3


def stiff(t, y, blk, thtrdat, impChem, i, j, k):
    blk.array["q"][i, j, k, 4::] = y
    impChem(blk, thtrdat, 10, i, j, k)

    return blk.array["omega"][i, j, k, :]


class strang:
    """
    Strang-splitting

    Second order accurate in time.

    Stiff reactions use implicit vode+bdf.
    Non-stiff transport use SSPRK3.

    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def non_stiff(self, dt):
        # First Stage
        self.titme = self.tme
        RHS(self)

        for blk in self:
            rk3s1(blk, dt)

        consistify(self)

        # Second Stage
        self.titme = self.tme + dt
        RHS(self)

        for blk in self:
            rk3s2(blk, dt)

        consistify(self)

        # Third Stage
        self.titme = self.tme + dt / 2.0
        RHS(self)

        for blk in self:
            rk3s3(blk, dt)

        consistify(self)

    def step(self, dt):
        ###############################################################
        # Take a half step in time for non-stiff operator
        ###############################################################
        dt /= 2.0

        self.non_stiff(dt)

        dt *= 2.0
        ###############################################################
        # Take a full step in time for stiff operator
        ###############################################################
        ODE = ode(stiff)
        ODE.set_integrator("vode", method="bdf", with_jacobian=True)
        for blk in self:
            it = product(
                range(blk.ng, blk.ni + blk.ng - 1),
                range(blk.ng, blk.nj + blk.ng - 1),
                range(blk.ng, blk.nk + blk.ng - 1),
            )
            for ijk in it:
                i, j, k = ijk

                y0 = blk.array["q"][i, j, k, 4::]
                ODE.set_initial_value(y0, 0.0)
                ODE.set_f_params(blk, self.thtrdat, self.impChem, i, j, k)
                ODE.integrate(dt)

                blk.array["Q"][i, j, k, 5::] = ODE.y[1::] * blk.array["Q"][i, j, k, 0]

        consistify(self)

        ###############################################################
        # Take final half step in time for non-stiff operator
        ###############################################################
        dt /= 2.0

        self.non_stiff(dt)

        dt *= 2.0
        ###############################################################
        # Step complete
        ###############################################################
        self.nrt += 1
        self.tme += dt
        self.titme = self.tme + dt

    step.name = "strang"
    step.stepType = "split"

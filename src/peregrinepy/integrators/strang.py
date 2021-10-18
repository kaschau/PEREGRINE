from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify
from scipy.integrate import ode
from itertools import product


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

    def stiff(t, y, mb, bindx, i, j, k):
        mb[bindx].array["q"][i, j, k, 4::] = y
        mb.impChem(mb[bindx], mb.thtrdat, 10, i, j, k)

        return mb[bindx].array["omega"][i, j, k, 0:-1]

    solver = ode(stiff)
    solver.set_integrator("vode", method="bdf", with_jacobian=True)

    def non_stiff(self, dt):
        # store zeroth stage solution
        for blk in self:
            blk.array["rhs0"][:] = blk.array["Q"][:]

        # First Stage
        RHS(self)

        for blk in self:
            blk.array["rhs1"][:] = dt * blk.array["dQ"]
            blk.array["Q"][:] = blk.array["rhs0"] + blk.array["rhs1"]

        consistify(self)

        # Second Stage
        RHS(self)

        for blk in self:
            blk.array["Q"][:] = (
                0.75 * blk.array["rhs0"]
                + 0.25 * blk.array["Q"]
                + 0.25 * blk.array["dQ"] * dt
            )

        consistify(self)

        # Third Stage
        RHS(self)

        for blk in self:
            blk.array["Q"][:] = (
                blk.array["rhs0"] + 2.0 * blk.array["Q"] + 2.0 * blk.array["dQ"] * dt
            ) / 3.0

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
        for bindx, blk in enumerate(self):
            it = product(
                range(blk.ng, blk.ni + blk.ng - 1),
                range(blk.ng, blk.nj + blk.ng - 1),
                range(blk.ng, blk.nk + blk.ng - 1),
            )
            for ijk in it:
                i, j, k = ijk

                y0 = blk.array["q"][i, j, k, 4::]
                self.solver.set_initial_value(y0, 0.0)
                self.solver.set_f_params(self, bindx, i, j, k)
                self.solver.integrate(dt)

                blk.array["Q"][i, j, k, 5::] = (
                    self.solver.y[1::] * blk.array["Q"][i, j, k, 0]
                )

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

    step.name = "strang"

from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify


class rk4:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def step(self, dt):

        # store zeroth stage solution
        for blk in self:
            blk.array["rhs0"][:] = blk.array["Q"][:]

        # First Stage

        RHS(self)

        for blk in self:
            blk.array["rhs1"][:] = dt * blk.array["dQ"]
            blk.array["Q"][:] = blk.array["rhs0"] + 0.5 * blk.array["rhs1"]

        consistify(self)

        # Second Stage

        RHS(self)

        for blk in self:
            blk.array["rhs2"][:] = dt * blk.array["dQ"]
            blk.array["Q"][:] = blk.array["rhs0"] + 0.5 * blk.array["rhs2"]

        consistify(self)

        # Third Stage

        RHS(self)

        for blk in self:
            blk.array["rhs3"][:] = dt * blk.array["dQ"]
            blk.array["Q"][:] = blk.array["rhs0"] + blk.array["rhs3"]

        consistify(self)

        # Fourth Stage

        RHS(self)

        for blk in self:
            blk.array["Q"][:] = (
                blk.array["rhs0"]
                + (
                    blk.array["rhs1"]
                    + 2.0 * blk.array["rhs2"]
                    + 2.0 * blk.array["rhs3"]
                    + dt * blk.array["dQ"]
                )
                / 6.0
            )

        consistify(self)

        self.nrt += 1
        self.tme += dt

    step.name = "rk4"

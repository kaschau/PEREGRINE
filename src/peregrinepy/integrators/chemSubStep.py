from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify
from ..compute.utils import AEQB, dQzero
from ..compute.timeIntegration import rk3s1, rk3s2, rk3s3


class chemSubStep:
    """
    Strang-splitting

    Second order accurate in time.

    Non-stiff transport use SSPRK3.
    Stiff chemistry takes n chemistry sub steps

    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def non_stiff(self, dt):
        # store zeroth stage solution
        for blk in self:
            AEQB(blk.rhs0, blk.Q)

        # First Stage
        RHS(self)

        for blk in self:
            rk3s1(blk, dt)

        consistify(self)

        # Second Stage
        RHS(self)

        for blk in self:
            rk3s2(blk, dt)

        consistify(self)

        # Third Stage
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

        n = 50
        for blk in self:
            for _ in range(n):
                # store zeroth stage solution
                AEQB(blk.rhs0, blk.Q)

                # First Stage
                dQzero(blk)
                self.impChem(blk, self.thtrdat)

                rk3s1(blk, dt / float(n))

                self.eos(blk, self.thtrdat, 0, "cons")

                # Second Stage
                dQzero(blk)
                self.impChem(blk, self.thtrdat)

                rk3s2(blk, dt / float(n))

                self.eos(blk, self.thtrdat, 0, "cons")

                # Third Stage
                dQzero(blk)
                self.impChem(blk, self.thtrdat)

                rk3s3(blk, dt / float(n))

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

    step.name = "chemSubStep"

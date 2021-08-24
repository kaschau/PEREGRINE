from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify

class rk1:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def step(self, dt):

        RHS(self)

        #add it to current solution
        for blk in self:
            blk.array['Q'][:] += dt*blk.array['dQ']

        self.nrt += 1
        self.tme += dt

        consistify(self)

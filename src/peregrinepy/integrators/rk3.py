from abc import ABCMeta
from ..RHS import RHS
from ..consistify import consistify

class rk3:
    '''
     S. Gottlieb, C.-W Shu. Total variation diminishing Runge-Kutta
     schemes. Mathematics of Computation, 67(221):73-85, 1998.

    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def step(self, dt):

        #store zeroth stage solution
        for blk in self:
            blk.array['rhs0'][:] = blk.array['Q'][:]

        # First Stage

        RHS(self)

        for blk in self:
            blk.array['rhs1'][:] = dt*blk.array['dQ']
            blk.array['Q'][:] = blk.array['rhs0'] + blk.array['rhs1']

        consistify(self)

        # Second Stage

        RHS(self)

        for blk in self:
            blk.array['Q'][:] =  0.75*blk.array['rhs0']   \
                               + 0.25*blk.array['Q']      \
                               + 0.25*blk.array['dQ']*dt

        consistify(self)

        # Third Stage

        RHS(self)

        for blk in self:
            blk.array['Q'][:] = (    blk.array['rhs0'] +    \
                                 2.0*blk.array['Q']    +    \
                                 2.0*blk.array['dQ']   *dt )/3.0

        consistify(self)

        self.nrt += 1
        self.tme += dt

from .bcs import apply_bcs
from .mpicomm.blockcomm import communicate
from .compute import dqdxyz

def consistify(mb):

    #We assume that the interior of the blocks have a
    # conservative Q variable field. We update the
    # interior primatives, apply boundary conditions,
    # update halo values as needed, then communicate
    # everything

    #Update interior primatives
    for blk in mb:
        mb.eos(blk,mb.thermdat,'0','cons')

    #Apply euler boundary conditions
    apply_bcs(mb,'euler')

    #communicate halos
    communicate(mb,['Q','q'])

    #Update spatial derivatives
    dqdxyz(mb)

    #Apply viscous boundary conditions
    apply_bcs(mb,'viscous')

    #communicate viscous halos
    communicate(mb,['dqdx','dqdy','dqdz'])

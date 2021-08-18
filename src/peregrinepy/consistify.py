from .bcs import apply_bcs
from .mpicomm.blockcomm import communicate

def consistify(mb):

    #We assume that the interior of the blocks have a
    # conservative Q variable field. We update the
    # interior primatives, apply boundary conditions,
    # update halo values as needed, then communicate
    # everything

    #Update interior primatives
    for blk in mb:
        mb.eos(blk,mb.thermdat,'0','cons')

    #Apply boundary conditions
    apply_bcs(mb)

    #communicate Q halos
    communicate(mb,'Q')
    communicate(mb,'q')

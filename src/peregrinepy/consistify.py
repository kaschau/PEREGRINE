from .bcs import apply_bcs
from .mpicomm.blockcomm import communicate
from .compute import momentum, cpg

def consistify(mb):

    #We assume that the interior of the blocks have a
    # conservative Q variable field. We update the
    # interior primatives, apply boundary conditions,
    # update halo values as needed, then communicate
    # everything

    #Update interior primatives
    for blk in mb:
        #Compute thermoprops, p,T,cp,h etc.
        cpg(blk,'0','Erho')
        #Compute u from rhou
        momentum(blk,'0','rhou')
        #Compute Y from rhoY
        momentum(blk,'0','rhou')

    #Apply boundary conditions
    apply_bcs(mb)

    #communicate Q halos
    communicate(mb,'Q')
    communicate(mb,'q')

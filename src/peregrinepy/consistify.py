from .bcs import apply_bcs
from .mpicomm.blockcomm import communicate
from .compute import momentum, EOS_ideal, calEOS_perfect

def consistify(mb):

    #We assume that the interior of the blocks have a
    # conservative Q variable field. We update the
    # interior primatives, apply boundary conditions,
    # update halo values as needed, then communicate
    # everything

    #Update interior primatives
    for blk in mb:
        #Compute T from E
        calEOS_perfect(blk,'0','Erho')
        #Compute P from T and rho
        EOS_ideal(blk,'0','rhoT')
        #Compute u from rhou
        momentum(blk,'0','rhou')


    #Apply boundary conditions
    apply_bcs(mb)

    #communicate Q halos
    communicate(mb,'Q')
    communicate(mb,'q')

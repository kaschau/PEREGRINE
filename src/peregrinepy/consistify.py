from .bcs import apply_bcs
from .mpicomm.blockcomm import communicate
from .compute import dqdxyz

def consistify(mb):

    #We assume that the interior of the blocks have a
    # conservative Q variable field. We update the
    # interior primatives, apply boundary conditions,
    # update halo values as needed, then communicate
    # everything

    #First communicate conservatives
    communicate(mb,['Q'])

    #Now update derived arrays for ENTIRE block, even exterior halos
    for blk in mb:
        mb.eos(blk,mb.thermdat,-1,'cons')

    #Apply euler boundary conditions
    apply_bcs(mb,'euler')

    if mb.config['RHS']['diffusion']:
        #Update spatial derivatives
        dqdxyz(mb)

        #Apply viscous boundary conditions
        apply_bcs(mb,'viscous')

        #communicate viscous halos
        communicate(mb,['dqdx','dqdy','dqdz'])

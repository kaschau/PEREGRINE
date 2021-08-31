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

    #Update interior primatives
    for blk in mb:
        mb.eos(blk,mb.thermdat,0,'cons')

    #Apply euler boundary conditions
    apply_bcs(mb,'euler')

    #now we need to update derived arrays for interior halos
    for blk in mb:
        for face in blk.faces:
            if face.connectivity['neighbor'] is not None:
                # update qh
                mb.eos(blk,mb.thermdat,face.nface,'cons')

    #communicate primatives
    communicate(mb,['q'])

    if mb.config['RHS']['diffusion']:
        #Update spatial derivatives
        dqdxyz(mb)

        #Apply viscous boundary conditions
        apply_bcs(mb,'viscous')

        #communicate viscous halos
        communicate(mb,['dqdx','dqdy','dqdz'])

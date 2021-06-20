from .bcs import apply_bcs

def consistify(mb, config):

    #We assume that the interior of the blocks have a
    # conservative Q variable field. We update the
    # interior primatives, apply boundary conditions,
    # update halo values as needed, then communicate
    # everything

    pass

    #communicate

from .mpiComm import communicate

"""
This function unifies the interior and halo cell data
such that interior data fields are thermodynamically
consistent, the halos match with neighbring blocks,
and the boundary conditions are set. Basically, make it
so we can compute the RHS values with up to date data.
"""


def consistify(mb, given="cons"):
    # We start with either conserved Q (given="cons"), or we
    # start with primative q (given="prims"). Then go from one
    # to the other, and then make the rest of pointwise arrays
    # thermodynamically consistent, fill in halos, and apply
    # boundary conditions.

    # First communicate conservatives/primatives
    if given == "cons":
        communicate(mb, ["Q"])
    elif given == "prims":
        communicate(mb, ["q"])

    # Now update derived arrays for ENTIRE block,
    # even exterior halos.
    for blk in mb:
        mb.eos(blk.cpp, mb.thtrdat.cpp, -1, given)

        # Apply euler boundary conditions
        for face in blk.faces:
            face.bcFunc(blk.cpp, face.cpp, mb.eos, mb.thtrdat.cpp, "euler", mb.titme)

        # Update transport properties
        mb.trans(blk.cpp, mb.thtrdat.cpp, -1)

        # Update switch
        mb.switch(blk.cpp)

        # Apply viscous sponge
        mb.viscousSponge(
            blk.cpp,
            mb.config["viscousSponge"]["origin"],
            mb.config["viscousSponge"]["ending"],
            mb.config["viscousSponge"]["multiplier"],
        )

    # Communicate necessary halos
    if mb.phiComm:
        communicate(mb, ["phi"])

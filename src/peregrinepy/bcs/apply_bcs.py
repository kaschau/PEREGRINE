from .inlets import *
from .exits import *
from .walls import *


def apply_bcs(mb, terms):

    # First we apply inlets and exits
    for blk in mb:
        for face in blk.faces:
            bc = face.connectivity["bctype"]
            if bc in ["b0", "b1"]:
                continue
            elif bc == "constant_velocity_subsonic_inlet":
                constant_velocity_subsonic_inlet(mb.eos, blk, face, mb.thtrdat, terms)
            elif bc == "constant_pressure_subsonic_exit":
                constant_pressure_subsonic_exit(mb.eos, blk, face, mb.thtrdat, terms)
    # Then we apply walls
    for blk in mb:
        for face in blk.faces:
            bc = face.connectivity["bctype"]
            if bc in ["b0", "b1"]:
                continue
            elif bc == "adiabatic_noslip_wall":
                adiabatic_noslip_wall(mb.eos, blk, face, mb.thtrdat, terms)
            elif bc == "adiabatic_slip_wall":
                adiabatic_slip_wall(mb.eos, blk, face, mb.thtrdat, terms)
            elif bc == "adiabatic_moving_wall":
                adiabatic_moving_wall(mb.eos, blk, face, mb.thtrdat, terms)
            elif bc == "isoT_moving_wall":
                isoT_moving_wall(mb.eos, blk, face, mb.thtrdat, terms)

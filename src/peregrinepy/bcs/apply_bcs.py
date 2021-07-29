from .inlets import subsonic_inlet
from .exits import subsonic_exit
from .walls import *

def apply_bcs(mb):

    #First we apply inlets and exits
    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            bc = blk.connectivity[face]['bc']
            if bc == 'i1':
                subsonic_inlet(blk,face,mb.thermdat)
            elif bc == 'e1':
                subsonic_exit(blk,face,mb.thermdat)
    # Then we apply walls
    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            bc = blk.connectivity[face]['bc']
            if bc == 's1':
                adiabatic_noslip_wall(blk,face,mb.thermdat)
            if bc == 's2':
                adiabatic_slip_wall(blk,face,mb.thermdat)

from .inlets import subsonic_inlet
from .exits import subsonic_exit
from .walls import *

def apply_bcs(mb,terms):

    #First we apply inlets and exits
    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            bc = blk.connectivity[face]['bc']
            if bc == 'i1':
                subsonic_inlet(mb.eos,blk,face,mb.thermdat,terms)
            elif bc == 'e1':
                subsonic_exit(mb.eos,blk,face,mb.thermdat,terms)
    # Then we apply walls
    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            bc = blk.connectivity[face]['bc']
            if bc == 's1':
                adiabatic_noslip_wall(mb.eos,blk,face,mb.thermdat,terms)
            if bc == 's2':
                adiabatic_slip_wall(mb.eos,blk,face,mb.thermdat,terms)
            if bc == 's3':
                adiabatic_moving_wall(mb.eos,blk,face,mb.thermdat,terms)

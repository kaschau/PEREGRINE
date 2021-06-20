from .inlets import subsonic_inlet
from .exits import subsonic_exit
from .walls import adiabatic_noslip_wall

def apply_bcs(mb,config):

    for blk in mb:
        for face in ['1','2','3','4','5','6']:
            bc = blk.connectivity[face]['bc']
            if bc == 's1':
                adiabatic_noslip_wall(blk,face)
            elif bc == 'i1':
                subsonic_inlet(blk,face)
            elif bc == 'e1':
                subsonic_exit(blk,face)
            elif bc.startswith('b'):
                pass
            else:
                raise ValueError(f'Unknown bc for blk #{blk.nblki} face {face}.')

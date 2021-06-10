from .blockcomm import communicate


def halo_grid(mb,config):

    communicate(mb,'x')
    communicate(mb,'y')
    communicate(mb,'z')

# -*- coding: utf-8 -*-

import yaml
from ..mpicomm import mpiutils

def read_connectivity(mb,path_to_file):
    '''This function parses a RAPTOR connectivity file given by file_path and adds the connectivity information to the supplied raptorpy.multiblock object

    Parameters
    ----------
    mb_data : raptorpy.multiblock.dataset (or a descendant)

    file_path : str
        Path to the conn.inp file to be read in

    Returns
    -------
    None
        Adds the connectivity information to mb

    '''
    comm,rank,size = mpiutils.get_comm_rank_size()

    #only the zeroth block reads in the file
    if 0 in mb.block_list:
        with open(f'{path_to_file}/bcs.yaml', 'r') as conn_file:
            bcs = yaml.load(conn_file, Loader=yaml.FullLoader)
    else:
        bcs = None
    bcs = comm.bcast(bcs, root=0)

    for blk in mb:
        for face in blk.faces:
            try:
                bcfam = face.connectivity['bcfam']
            except KeyError:
                print(f'Warning, block {blk.nblki} face {face.nface} is assigned the bcfam {bcfam} however that family is not defined in bcs.yaml')
            if bcfam != None:
                for key in bcs['bcfam'][key]:
                    face.bc[key] = bcs['bcfam'][key]
                face.bc._freeze()

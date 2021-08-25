# -*- coding: utf-8 -*-

import yaml
from ..mpicomm import mpiutils

def read_bcs(mb,path_to_file):
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
        try:
            with open(f'{path_to_file}/bcfams.yaml', 'r') as conn_file:
                bcs = yaml.load(conn_file, Loader=yaml.FullLoader)
        except IOError:
            bcs = None
    else:
        bcs = None
    bcs = comm.bcast(bcs, root=0)

    if bcs is None:
        return

    for blk in mb:
        for face in blk.faces:
            try:
                bcfam = face.connectivity['bcfam']
            except KeyError:
                print(f'Warning, block {blk.nblki} face {face.nface} is assigned the bcfam {bcfam} however that family is not defined in bcs.yaml')
            if bcfam is None:
                continue

            #Make sure the type in the input file matches the type in the connectivity
            if bcs[bcfam]['bctype'] != face.connectivity['bctype']:
                raise KeyError(f'Warning, block {blk.nblki} face {face.nface} does not match the bctype between input *{bcs[bcfam]["bctype"]}* and connectivity *{face.connectivity["bctype"]}*.')
            for key in bcs[bcfam].keys():
                face.bc[key] = bcs[bcfam][key]

            face.bc._freeze()

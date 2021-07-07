# -*- coding: utf-8 -*-

import json

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

    with open(f'{path_to_file}/conn.json', 'r') as conn_file:
        conn = json.load(conn_file)
    if mb is None:
        return conn
    else:
        nblks = len(conn)
        if nblks != mb.nblks:
            raise ValueError('WARNING! Number of blocks in dataset does not equal number of blocks in connectivity file.')
        for blk in mb:
            for face in ['1','2','3','4','5','6']:
                for k1 in conn[blk.nblki][face].keys():
                    blk.connectivity[face][k1] = conn[blk.nblki][face][k1]

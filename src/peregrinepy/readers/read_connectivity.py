# -*- coding: utf-8 -*-

import json

def read_connectivity(config):
    '''This function parses a RAPTOR connectivity file given by file_path and adds the connectivity information to the supplied raptorpy.multiblock object

    Parameters
    ----------
    mb_data : raptorpy.multiblock.dataset (or a descendant)

    file_path : str
        Path to the conn.inp file to be read in

    Returns
    -------
    None
        Adds the connectivity information to the mb_data parameter

    '''

    with open(f"{config['io']['inputdir']}/conn.json", 'r') as conn_file:

        conn = json.load(conn_file)

    return conn

        #nblks = int(conn_file.readline().split()[0])

        #if nblks != mb.nblks:
        #    raise ValueError('WARNING! Number of blocks in dataset does not equal number of blocks in connectivity file.')

        #lines = conn_file.readlines()
        #for blk,line in zip(mb,lines):
        #    temp = line.strip().split()[2::]
        #    for key in blk.connectivity.keys():
        #        face_data = temp[(int(key)-1)*4:(int(key)-1)*4+4]
        #        blk.connectivity[key]['bc'] = '{} {}'.format(face_data[0],face_data[1])
        #        blk.connectivity[key]['neighbor'] = face_data[2]
        #        blk.connectivity[key]['orientation'] = face_data[3]

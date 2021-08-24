# -*- coding: utf-8 -*-
import yaml

def write_connectivity(mb, path='./'):
    '''This function produces RAPTOR grid connectivity file (conn.inp) files from a raptorpy.multiblock.grid (or a descendant)

    Parameters
    ----------

    mb : raptorpy.multiblock.grid (or a descendant)

    file_path : str
       Path of location to write output RAPTOR grid connectivity file to

    Returns
    -------
    None


    '''

    conn = {}
    conn['Total_Blocks'] = len(mb)
    for blk in mb:
        conn[f'Block{blk.nblki}'] = blk.connectivity()

    with open(f"{path}/conn.yaml", 'w') as f:
        yaml.dump(conn, f, sort_keys=False)

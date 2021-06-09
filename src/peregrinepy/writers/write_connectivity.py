# -*- coding: utf-8 -*-
import json

def write_connectivity(mb, config):
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

    conn = []
    for blk in mb:
        conn.append(blk.connectivity)

    with open(f"{config['io']['inputdir']}/conn.json", 'w') as conn_file:

        conn_file.write(json.dumps(conn, indent=4))

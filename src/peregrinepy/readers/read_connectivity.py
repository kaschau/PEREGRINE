# -*- coding: utf-8 -*-

import yaml
from ..mpicomm import mpiutils


def read_connectivity(mb, path_to_file):
    """This function parses a RAPTOR connectivity file given by file_path and adds the connectivity information to the supplied raptorpy.multiblock object

    Parameters
    ----------
    mb_data : raptorpy.multiblock.dataset (or a descendant)

    file_path : str
        Path to the conn.inp file to be read in

    Returns
    -------
    None
        Adds the connectivity information to mb

    """
    comm, rank, size = mpiutils.get_comm_rank_size()

    # only the zeroth rank reads in the file
    if rank == 0:
        with open(f"{path_to_file}/conn.yaml", "r") as conn_file:
            conn = yaml.load(conn_file, Loader=yaml.FullLoader)
    else:
        conn = None
    conn = comm.bcast(conn, root=0)

    for blk in mb:
        myconn = conn[f"Block{blk.nblki}"]
        for face in blk.faces:
            for k1 in face.connectivity.keys():
                val = myconn[f"Face{face.nface}"][k1]
                face.connectivity[k1] = val

    mb.total_blocks = conn["Total_Blocks"]

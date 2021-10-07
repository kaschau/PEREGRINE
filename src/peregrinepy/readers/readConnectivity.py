# -*- coding: utf-8 -*-

import yaml
from ..mpiComm import mpiUtils


def readConnectivity(mb, pathToFile):
    """
    This function parses a PEREGRINE connectivity file given
    by pathToFile and adds the connectivity information to
    the supplied peregrinepy.multiBlock object

    Parameters
    ----------
    mb : peregrine.multiBlock.topology (or a descendant)

    pathToFile : str
        Path to the conn.yaml file to be read in

    Returns
    -------
    None
        Adds the connectivity information to mb

    """
    comm, rank, size = mpiUtils.getCommRankSize()

    # only the zeroth rank reads in the file
    if rank == 0:
        with open(f"{pathToFile}/conn.yaml", "r") as connFile:
            conn = yaml.load(connFile, Loader=yaml.FullLoader)
    else:
        conn = None
    conn = comm.bcast(conn, root=0)

    for blk in mb:
        myConn = conn[f"Block{blk.nblki}"]
        for face in blk.faces:
            for k1 in face.connectivity.keys():
                val = myConn[f"Face{face.nface}"][k1]
                face.connectivity[k1] = val

    mb.totalBlocks = conn["Total_Blocks"]

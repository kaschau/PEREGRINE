# -*- coding: utf-8 -*-

import yaml


def readConnectivity(mb, pathToFile, parallel=False):
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

    if parallel:
        from ..mpiComm import mpiUtils

        comm, rank, size = mpiUtils.getCommRankSize()

        # only the zeroth rank reads in the file
        if rank == 0:
            with open(f"{pathToFile}/conn.yaml", "r") as connFile:
                conn = yaml.load(connFile, Loader=yaml.FullLoader)
        else:
            conn = None
        conn = comm.bcast(conn, root=0)

    else:
        with open(f"{pathToFile}/conn.yaml", "r") as connFile:
            conn = yaml.load(connFile, Loader=yaml.FullLoader)

    for blk in mb:
        myConn = conn[f"Block{blk.nblki}"]
        for face in blk.faces:
            fdict = myConn[f"Face{face.nface}"]
            face.bcType = fdict["bcType"]
            face.bcFam = fdict["bcFam"]
            face.neighbor = fdict["neighbor"]
            face.orientation = fdict["orientation"]

            try:
                face.isPeriodicLow = fdict["isPeriodicLow"]
                assert type(face.isPeriodicLow) is bool
            except KeyError:
                pass

    mb.totalBlocks = conn["Total_Blocks"]

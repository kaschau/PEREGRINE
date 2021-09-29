# -*- coding: utf-8 -*-

import yaml
from ..mpicomm import mpiutils


def readBcs(mb, pathToFile):
    """
    This function parses a RAPTOR connectivity file given by
    file_path and adds the connectivity information to the
    supplied raptorpy.multiblock object

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
    comm, rank, size = mpiutils.getCommRankSize()

    # only the zeroth block reads in the file
    if 0 in mb.block_list:
        try:
            with open(f"{pathToFile}/bcFams.yaml", "r") as connFile:
                bcsIn = yaml.load(connFile, Loader=yaml.FullLoader)
        except IOError:
            bcsIn = None
    else:
        bcsIn = None
    bcsIn = comm.bcast(bcsIn, root=0)

    if bcsIn is None:
        if rank == 0:
            print("No bcFams.yaml found, using defaults.")
        return

    for blk in mb:
        for face in blk.faces:
            try:
                bcFam = face.connectivity["bcFam"]
            except KeyError:
                print(
                    f"Warning, block {blk.nblki} face {face.nface} is assigned the bcFam {bcFam} however that family is not defined in bcFams.yaml"
                )
            if bcFam is None:
                continue

            # Make sure the type in the input file matches the type in the connectivity
            if bcsIn[bcFam]["bcType"] != face.connectivity["bcType"]:
                raise KeyError(
                    f'Warning, block {blk.nblki} face {face.nface} does not match the bcType between input *{bcsIn[bcFam]["bcType"]}* and connectivity *{face.connectivity["bcType"]}*.'
                )

            # Set the boundary condition values
            for key in bcsIn[bcFam]["values"]:
                face.bcVals[key] = bcsIn[bcFam]["values"][key]

            # Now add any un-specified species as zero (unless we have single component)
            if blk.ns != 1:
                for sn in blk.speciesNames[0:-1]:
                    if sn not in face.bcVals.keys():
                        face.bcVals[sn] = 0.0
            else:
                face.bcVals[mb.speciesNames[0]] = 1.0

            face.bcVals._freeze()

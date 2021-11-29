# -*- coding: utf-8 -*-

import numpy as np
import yaml
from ..mpiComm import mpiUtils
from .. import bcs


def readBcs(mb, pathToFile):
    """
    This function parses a RAPTOR connectivity file given by
    file_path and adds the connectivity information to the
    supplied peregrinepy.multiBlock object

    Parameters
    ----------
    mb_data : peregrinepy.multiBlock.dataset (or a descendant)

    file_path : str
        Path to the conn.inp file to be read in

    Returns
    -------
    None
        Adds the connectivity information to mb

    """
    comm, rank, size = mpiUtils.getCommRankSize()

    # only the zeroth block reads in the file
    try:
        with open(f"{pathToFile}/bcFams.yaml", "r") as connFile:
            bcsIn = yaml.load(connFile, Loader=yaml.FullLoader)
    except IOError:
        bcsIn = None

    if bcsIn is None:
        if rank == 0:
            print("No bcFams.yaml found, using defaults.")
        return

    for blk in mb:
        for face in blk.faces:
            bcFam = face.bcFam
            if bcFam is None:
                continue

            # Make sure the type in the input file matches the type in the connectivity
            if bcsIn[bcFam]["bcType"] != face.bcType:
                raise KeyError(
                    f'ERROR, block {blk.nblki} face {face.nface} does not match the bcType between input *{bcsIn[bcFam]["bcType"]}* and connectivity *{face.bcType}*.'
                )

            # Set the boundary condition values
            face.array["qBcVals"] = np.zeros(blk.array["q"][face.s1_].shape)
            face.array["QBcVals"] = np.zeros(blk.array["Q"][face.s1_].shape)

            # If there are no values to set, continue
            if "bcVals" not in bcsIn[bcFam]:
                print(f"Warning, no values found for {bcsIn[bcFam]}")
                continue

            # Certain boundary conditions need prep work,
            # such as constant mass or profiles, so call them here
            inputValues = bcsIn[bcFam]["bcVals"]
            for bcmodule in [bcs.prepInlets, bcs.prepExits, bcs.prepWalls]:
                try:
                    func = getattr(bcmodule, "prep_" + face.bcType)
                    func(blk, face, inputValues)
                    break
                except AttributeError:
                    pass
            else:
                raise ValueError(f"Could not find the prep_ function for {face.bcType}")

            # If we are a solver face, we need to create the kokkos arrays
            if face.faceType == "solver":
                from ..misc import createViewMirrorArray

                space = mb.config["Kokkos"]["Space"]

                names = ["qBcVals", "QBcVals"]
                shape = blk.array["q"][face.s1_].shape
                createViewMirrorArray(face, names, shape, space)

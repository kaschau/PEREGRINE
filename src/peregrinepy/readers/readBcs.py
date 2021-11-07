# -*- coding: utf-8 -*-

import yaml
from ..mpiComm import mpiUtils
import numpy as np


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
                bcFam = face.bcFam
            except KeyError:
                print(
                    f"Warning, block {blk.nblki} face {face.nface} is assigned the bcFam {bcFam} however that family is not defined in bcFams.yaml"
                )
            if bcFam is None:
                continue

            # Make sure the type in the input file matches the type in the connectivity
            if bcsIn[bcFam]["bcType"] != face.bcType:
                raise KeyError(
                    f'ERROR, block {blk.nblki} face {face.nface} does not match the bcType between input *{bcsIn[bcFam]["bcType"]}* and connectivity *{face.bcType}*.'
                )

            # Set the boundary condition values
            face.array["qBcVals"] = np.zeros(5 + blk.ns - 1)
            face.array["QBcVals"] = np.zeros(5 + blk.ns - 1)
            qIndexMap = {"p": 0, "u": 1, "v": 2, "w": 3, "T": 4}
            QIndexMap = {"rho": 0, "rhou": 1, "rhov": 2, "rhow": 3, "rhoE": 4}
            for i in range(blk.ns):
                qIndexMap[blk.speciesNames[i]] = 5 + i
                QIndexMap["rho" + blk.speciesNames[i]] = 5 + i

            if "bcVals" in bcsIn[bcFam].keys():
                for key in bcsIn[bcFam]["bcVals"]:
                    if key in qIndexMap.keys():
                        indx = qIndexMap[key]
                        face.array["qBcVals"][indx] = float(bcsIn[bcFam]["bcVals"][key])
                    elif key in QIndexMap.keys():
                        face.array["QBcVals"][indx] = float(bcsIn[bcFam]["bcVals"][key])
                    else:
                        raise KeyError("ERROR, unknown input bcVal: {key}")

            # If we are a solver face, we need to create the kokkos arrays
            if face.faceType == "solver":
                import kokkos

                if mb.config["Kokkos"]["Space"] in ["OpenMP", "Serial", "Default"]:
                    space = kokkos.HostSpace
                elif mb.config["Kokkos"]["Space"] in ["Cuda"]:
                    space = kokkos.CudaSpace
                else:
                    raise ValueError("What space?")

                for name in ["qBcVals", "QBcVals"]:
                    setattr(
                        face,
                        name,
                        kokkos.array(
                            face.array[name],
                            dtype=kokkos.double,
                            space=space,
                            dynamic=False,
                        ),
                    )

import numpy as np
import yaml
from .. import bcs


def readBcs(mb, pathToFile, justPeriodic=False):
    """
    This function parses a PEREGRINE bcFam.yaml file given by
    pathToFile and adds the relevant conditions to the boundary
    conditions of the block faces.

    Parameters
    ----------
    mb : peregrinepy.multiBlock.topology (or a descendant)

    pathToFile : str
        Path to the bcFam.yaml file to be read in

    justPeriodic : bool
        For runnning cases, we need periodic boundary info
        BEFORE we unify the grid, but we need grid metrics
        for some of the boundary condition types. So we want
        to be able to read in just periodic info sometimes.

    Returns
    -------
    None
        Adds the connectivity information to mb

    """

    # only the zeroth block reads in the file
    try:
        with open(f"{pathToFile}/bcFams.yaml", "r") as connFile:
            bcsIn = yaml.load(connFile, Loader=yaml.FullLoader)
    except IOError:
        bcsIn = None

    if bcsIn is None:
        if 0 in mb.blockList:
            print("No bcFams.yaml found, assuming all defaults.")
        return

    warn = []
    for blk in mb:
        for face in blk.faces:
            bcFam = face.bcFam
            bcType = face.bcType
            # Not all bc types need inputs from bcFam, so we check here
            if bcFam is None:
                if bcType not in (
                    "b0",
                    "supersonicExit",
                    "adiabaticNoSlipWall",
                    "adiabaticSlipWall",
                ):
                    raise ValueError(
                        f"bcType {bcType} in block {blk.nblki}, face {face.nface} requires a bcFam."
                    )
                continue

            # Make sure the type in the input file matches the type in the connectivity
            # Because periodics have a high/low designation they wont match exactly
            if bcsIn[bcFam]["bcType"] != face.bcType:
                if bcsIn[bcFam]["bcType"].startswith("periodic"):
                    if not face.bcType.startswith(bcsIn[bcFam]["bcType"]):
                        raise KeyError(
                            f'ERROR, block {blk.nblki} face {face.nface} does not match the bcType between input *{bcsIn[bcFam]["bcType"]}* and connectivity *{face.bcType}*.'
                        )
                else:
                    raise KeyError(
                        f'ERROR, block {blk.nblki} face {face.nface} does not match the bcType between input *{bcsIn[bcFam]["bcType"]}* and connectivity *{face.bcType}*.'
                    )

            # If there are no values to set, continue
            if "bcVals" not in bcsIn[bcFam]:
                if bcsIn[bcFam] not in warn:
                    print(f"Warning, no values found for {bcsIn[bcFam]}")
                    warn.append(bcsIn[bcFam])
                continue

            # Add the information from any periodic faces
            if bcType.startswith("periodic"):
                face.periodicSpan = bcsIn[bcFam]["bcVals"]["periodicSpan"]
                face.periodicAxis = bcsIn[bcFam]["bcVals"]["periodicAxis"]
                continue
            if justPeriodic:
                continue

            # If we are a solver face, we need to create the kokkos arrays
            if face.faceType != "solver":
                continue

            from ..misc import createViewMirrorArray

            # Create "profile" arrays for bc values
            face.array["qBcVals"] = np.zeros(blk.array["q"][face.s1_].shape)
            face.array["QBcVals"] = np.zeros(blk.array["Q"][face.s1_].shape)

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

            names = ["qBcVals", "QBcVals"]
            shape = blk.array["q"][face.s1_].shape
            createViewMirrorArray(face, names, shape)

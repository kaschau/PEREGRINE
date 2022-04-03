# -*- coding: utf-8 -*-
import yaml


def writeConnectivity(mb, path="./"):
    """This function produces PEREGRINE grid connectivity file (conn.yaml) files from
    a peregrinepy.multiBlock.topology (or a descendant)

    Parameters
    ----------

    mb : peregrinepy.multiBlock.grid (or a descendant)

    file_path : str
       Path of location to write output RAPTOR grid connectivity file to

    Returns
    -------
    None


    """

    conn = {}
    conn["Total_Blocks"] = len(mb)
    for blk in mb:
        conn[f"Block{blk.nblki}"] = {}
        bdict = conn[f"Block{blk.nblki}"]
        for face in blk.faces:
            bdict[f"Face{face.nface}"] = {}
            fdict = bdict[f"Face{face.nface}"]
            fdict["bcType"] = face.bcType
            fdict["bcFam"] = face.bcFam
            fdict["neighbor"] = face.neighbor
            fdict["orientation"] = face.orientation
            if face.isPeriodicLow is not None:
                fdict["isPeriodicLow"] = face.isPeriodicLow

    with open(f"{path}/conn.yaml", "w") as f:
        yaml.dump(conn, f, sort_keys=False)

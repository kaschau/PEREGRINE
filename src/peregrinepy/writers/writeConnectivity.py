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
        for fc in blk.faces:
            bdict[f"Face{fc.nface}"] = {}
            fdict = bdict[f"Face{fc.nface}"]
            fdict["bcType"] = fc.bcType
            fdict["bcFam"] = fc.bcFam
            fdict["neighbor"] = fc.neighbor
            fdict["orientation"] = fc.orientation

    with open(f"{path}/conn.yaml", "w") as f:
        yaml.dump(conn, f, sort_keys=False)

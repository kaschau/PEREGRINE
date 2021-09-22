from .thtrdat import thtrdat
from .MM_Tables import *


def get_eos(eos):

    if eos == "cpg":
        from ..compute.thermo import cpg

        return cpg
    elif eos == "tpg":
        from ..compute.thermo import tpg

        return tpg
    else:
        raise ValueError("What EOS?")

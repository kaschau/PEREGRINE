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


def get_trans(trans):

    if trans == "kineticTheory":
        from ..compute.transport import kineticTheory
        return kineticTheory
    else:
        raise ValueError("What transport method?")

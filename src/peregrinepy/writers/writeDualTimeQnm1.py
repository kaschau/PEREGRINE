# -*- coding: utf-8 -*-

import numpy as np


def writeDualTimeQnm1(mb, path="./", animate=True):
    # Save dualTime Qnm1 array
    for blk in mb:
        blk.updateHostView(["Qnm1"])
        if animate:
            fileName = f"{path}/Qnm1.{mb.nrt:08d}.{blk.nblki:06d}.npy"
        else:
            fileName = f"{path}/Qnm1.{blk.nblki:06d}.npy"
        with open(fileName, "wb") as f:
            np.save(f, blk.array["Qnm1"])

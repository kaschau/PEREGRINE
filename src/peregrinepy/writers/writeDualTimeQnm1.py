import numpy as np


def writeDualTimeQnm1(mb, path="./"):
    # Save dualTime Qnm1 array
    for blk in mb:
        ng = blk.ng
        blk.updateHostView(["Qnm1"])
        fileName = f"{path}/Qnm1.{mb.nrt:08d}.{blk.nblki:06d}.npy"
        with open(fileName, "wb") as f:
            np.save(f, blk.array["Qnm1"][ng:-ng, ng:-ng, ng:-ng, :])

from ..mpiComm.mpiUtils import getCommRankSize
from .base import BasePlugin


class Report(BasePlugin):
    """The step just taken: its number, time, size and CFL numbers. The CFL
    is reduced over the ranks here, so a run pays for it only when it asks."""

    name = "report"

    def __call__(self, solver):
        acoustic, convective, both = solver.maxCFL()
        if getCommRankSize()[1] != 0:
            return
        dt = solver.dt
        print(
            f" >>> --------- nrt: {solver.nrt:<6} ---------- <<<\n",
            f"    tme: {solver.tme:.6E} s\n"
            f"     dt : {dt:.6E} s\n"
            f"     MAX CFL       : {both * dt:.3f}\n"
            f"         Acoustic  : {acoustic * dt:.3f}\n"
            f"         Convective: {convective * dt:.3f}\n"
            " >>> -------------------------------- <<<\n",
        )

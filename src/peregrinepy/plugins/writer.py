import os
from pathlib import Path

from ..mpiComm.mpiUtils import getCommRankSize
from ..writers import GridWriter, RestartWriter
from .base import BasePlugin


class Writer(BasePlugin):
    """Results into a directory, made if need be, named from the step n or
    the time t (`basename`, q.{n:08d} unless said), carrying whatever the
    stepper keeps beyond the state. The xdmf points at the
    grid file the case came from; a case meshed in a script gets one written
    into the directory first."""

    name = "writer"

    def __init__(self, solver, cfgsect):
        super().__init__(solver, cfgsect)
        comm, rank, size = getCommRankSize()
        self.dir = Path(cfgsect.get("dir", "."))
        if rank == 0:
            self.dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        precision = cfgsect.get("precision", "single")
        gridFile = solver.meshFile
        if gridFile is None:
            GridWriter(solver, str(self.dir), precision).write(solver)
            gridFile = self.dir / "g.h5"
        gridDir = os.path.relpath(Path(gridFile).parent, self.dir)
        basename = cfgsect.get("basename", "q.{n:08d}")
        self.restart = RestartWriter(
            solver,
            str(self.dir),
            gridDir,
            precision,
            basename=basename,
            extras=solver.restartArrays,
            config=solver.config,
        )

    def __call__(self, solver):
        self.restart.write(solver)

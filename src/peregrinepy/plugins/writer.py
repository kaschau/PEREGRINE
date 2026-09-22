import os
from pathlib import Path

from ..misc import getCommRankSize
from ..writers import GridWriter, RestartWriter
from .base import BasePlugin


class Writer(BasePlugin):
    """Writes a result every time it is due, into the directory its section
    names."""

    name = "writer"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.cfgsect = cfgsect

    def start(self, solver):
        cfgsect = self.cfgsect
        comm, rank, size = getCommRankSize()
        self.dir = Path(cfgsect.get("dir", "."))
        if rank == 0:
            self.dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        precision = cfgsect.get("precision", "single")
        gridFile = solver.mesh.fileName
        if gridFile is None:
            gridFile = self.dir / "g.h5"
            GridWriter(solver, str(gridFile), precision).write(solver)
        basename = cfgsect.get("basename", "q.{n:08d}")
        self.restart = RestartWriter(
            solver,
            str(self.dir / f"{basename}.h5"),
            os.path.relpath(gridFile, self.dir),
            precision,
            extras=solver.integrator.restartArrays,
            config=solver.config,
        )

    def __call__(self, solver):
        self.restart.write(solver)

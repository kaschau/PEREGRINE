import numpy as np

from .base import BasePlugin


class Catalyst(BasePlugin):
    """ParaView Catalyst in situ processing, driven by the script the config
    names."""

    name = "catalyst"

    def __init__(self, mb, cfgsect):
        super().__init__(mb, cfgsect)
        # only a case that asks for catalyst needs paraview importable
        from paraview import vtk
        from paraview.catalyst import bridge
        from paraview.modules import vtkPVCatalyst as catalyst
        from paraview.vtk.util import numpy_support

        self.vtk, self.bridge, self.catalyst = vtk, bridge, catalyst
        self.numpy_support = numpy_support
        bridge.initialize()
        bridge.add_pipeline(cfgsect["script"])

        self._coProcessor = bridge.coprocessor

        # Save the data descriptions
        self.dataDescription = self.catalyst.vtkCPDataDescription()
        # Add the input input
        self.dataDescription.AddInput("input")

        # Create the multiblockdataset
        mbds = self.vtk.vtkMultiBlockDataSet()
        mbds.SetNumberOfBlocks(mb.totalBlocks)
        for i in range(mb.totalBlocks):
            mbds.SetBlock(i, None)

        # Create the grid and data arrays
        for blk in mb.blocks:
            ng = blk.ng
            q, Q = blk.q.get(), blk.Q.get()
            grid = self.vtk.vtkStructuredGrid()
            grid.SetDimensions(blk.ni, blk.nj, blk.nk)
            interior = blk.hostCopy("nodes")[ng:-ng, ng:-ng, ng:-ng]
            coords = np.column_stack(
                [interior[..., n].ravel(order="F") for n in range(3)]
            )
            points = self.vtk.vtkPoints()
            points.SetData(self.numpy_support.numpy_to_vtk(coords))
            grid.SetPoints(points)

            # density arrays
            self.addArray(grid, "rho", Q[ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F"))

            # pressure arrays
            self.addArray(
                grid,
                "p",
                q[ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F"),
            )

            # velocity array
            array = np.column_stack(
                tuple(
                    [q[ng:-ng, ng:-ng, ng:-ng, i].ravel(order="F") for i in (1, 2, 3)]
                )
            )
            self.addArray(grid, "Velocity", array)

            # temperature arrays
            self.addArray(
                grid,
                "T",
                q[ng:-ng, ng:-ng, ng:-ng, 4].ravel(order="F"),
            )

            for n, var in enumerate(blk.speciesNames[0:-1]):
                self.addArray(
                    grid,
                    var,
                    q[ng:-ng, ng:-ng, ng:-ng, 5 + n].ravel(order="F"),
                )

            # Add nth species
            array = self.numpy_support.numpy_to_vtk(
                1.0 - np.sum(q[ng:-ng, ng:-ng, ng:-ng, 5::], axis=-1).ravel(order="F")
            )
            self.addArray(grid, blk.speciesNames[-1], array)

            mbds.SetBlock(blk.nblki, grid)

        self.dataDescription.GetInputDescriptionByName("input").SetGrid(mbds)

    def addArray(self, grid, arrayName, npArray):
        # convert incoming numpy array to vtk
        vtkArray = self.numpy_support.numpy_to_vtk(npArray)
        vtkArray.SetName(arrayName)
        grid.GetCellData().AddArray(vtkArray)

    def swapArray(self, grid, arrayName, npArray):
        grid.GetCellData().RemoveArray(arrayName)
        # convert incoming numpy array to vtk
        vtkArray = self.numpy_support.numpy_to_vtk(npArray)
        vtkArray.SetName(arrayName)
        grid.GetCellData().AddArray(vtkArray)

    def __call__(self, mb):
        self.dataDescription.SetTimeData(mb.tme, mb.nrt)

        if not self._coProcessor.RequestDataDescription(self.dataDescription):
            return

        mbds = self.dataDescription.GetInputDescriptionByName("input").GetGrid()
        for blk in mb.blocks:
            q, Q = blk.q.get(), blk.Q.get()
            ng = blk.ng
            grid = mbds.GetBlock(blk.nblki)

            # density array
            self.swapArray(grid, "rho", Q[ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F"))

            # pressure array
            self.swapArray(grid, "p", q[ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F"))

            # velocity array
            array = np.column_stack(
                tuple(
                    [q[ng:-ng, ng:-ng, ng:-ng, i].ravel(order="F") for i in (1, 2, 3)]
                )
            )
            self.addArray(grid, "Velocity", array)

            # temperature array
            self.swapArray(grid, "T", q[ng:-ng, ng:-ng, ng:-ng, 4].ravel(order="F"))

            # species arrrays
            for n, var in enumerate(blk.speciesNames[0:-1]):
                self.swapArray(
                    grid,
                    var,
                    q[ng:-ng, ng:-ng, ng:-ng, 5 + n].ravel(order="F"),
                )

            # Add nth species
            array = 1.0 - np.sum(q[ng:-ng, ng:-ng, ng:-ng, 5::], axis=-1).ravel(
                order="F"
            )
            self.swapArray(grid, blk.speciesNames[-1], array.ravel(order="F"))

        # Execute coprocessing
        self._coProcessor.CoProcess(self.dataDescription)

    def finalize(self, mb):
        self.bridge.finalize()

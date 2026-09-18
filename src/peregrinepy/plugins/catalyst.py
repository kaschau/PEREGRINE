import numpy as np

from .base import BasePlugin


class Catalyst(BasePlugin):
    """ParaView Catalyst in situ processing, driven by the script the config
    names."""

    name = "catalyst"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.cfgsect = cfgsect

    def start(self, mb):
        cfgsect = self.cfgsect
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
            grid = self.vtk.vtkStructuredGrid()
            grid.SetDimensions(blk.ni, blk.nj, blk.nk)
            interior = blk.nodes.get()[ng:-ng, ng:-ng, ng:-ng]
            coords = np.column_stack(
                [interior[..., n].ravel(order="F") for n in range(3)]
            )
            points = self.vtk.vtkPoints()
            points.SetData(self.numpy_support.numpy_to_vtk(coords))
            grid.SetPoints(points)
            for name, array in self._arrays(mb, blk).items():
                self.addArray(grid, name, array)
            mbds.SetBlock(blk.nblki, grid)

        self.dataDescription.GetInputDescriptionByName("input").SetGrid(mbds)

    def _arrays(self, mb, blk):
        """Gives every export variable of a block over its interior, flat in
        the grid's order, the velocity components as one vector."""
        ng = blk.ng
        flat = lambda a: a[ng:-ng, ng:-ng, ng:-ng].ravel(order="F")
        values = mb.exportData(blk, mb.exportVars)
        arrays = {n: flat(a) for n, a in values.items() if n not in "uvw"}
        arrays["Velocity"] = np.column_stack([flat(values[n]) for n in "uvw"])
        return arrays

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
            grid = mbds.GetBlock(blk.nblki)
            for name, array in self._arrays(mb, blk).items():
                self.swapArray(grid, name, array)

        # Execute coprocessing
        self._coProcessor.CoProcess(self.dataDescription)

    def finalize(self, mb):
        self.bridge.finalize()

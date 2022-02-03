import paraview
from paraview.modules.vtkPVCatalyst import vtkCPProcessor, vtkCPDataDescription
from paraview.vtk.util import numpy_support

try:
    from paraview.modules.vtkPVPythonCatalyst import (
        vtkCPPythonScriptV2Pipeline as vtkPipeline,
    )
except ImportError:
    from paraview.modules.vtkPVPythonCatalyst import (
        vtkCPPythonScriptPipeline as vtkPipeline,
    )

from paraview.modules.vtkRemotingCore import vtkProcessModule
import vtk
import numpy as np


class coprocessor:
    def __init__(self, mb):

        # Try and figure out if we are using paraview > or < 5.9

        # Sanity check
        pm = vtkProcessModule.GetProcessModule()
        if pm and pm.GetPartitionId() == 0:
            print(
                "Warning: ParaView has been initialized before `initialize` is called"
            )

        # Initialize
        paraview.options.batch = True
        paraview.options.symmetric = True

        self._coProcessor = vtkCPProcessor()
        if not self._coProcessor.Initialize():
            raise RuntimeError("Failed to initialize Catalyst")

        # Add the coproc script
        fileName = mb.config["Catalyst"]["cpFile"]
        pipeline = vtkPipeline()
        if not pipeline.Initialize(fileName):
            raise RuntimeError("pipeline nitialization failed!")
        self._coProcessor.AddPipeline(pipeline)

        # Save the data descriptions
        self.dataDescription = vtkCPDataDescription()
        # Add the input input
        self.dataDescription.AddInput("input")

        # Create the multiblockdataset
        mbds = vtk.vtkMultiBlockDataSet()
        mbds.SetNumberOfBlocks(mb.totalBlocks)
        for i in range(mb.totalBlocks):
            mbds.SetBlock(i, None)

        # Create the grid and data arrays
        for blk in mb:
            ng = blk.ng
            grid = vtk.vtkStructuredGrid()
            grid.SetDimensions(blk.ni, blk.nj, blk.nk)
            coords = np.column_stack(
                tuple(
                    [
                        blk.array[var][ng:-ng, ng:-ng, ng:-ng].ravel(order="F")
                        for var in ("x", "y", "z")
                    ]
                )
            )
            points = vtk.vtkPoints()
            points.SetData(numpy_support.numpy_to_vtk(coords))
            grid.SetPoints(points)

            # density arrays
            self.addArray(
                grid, "rho", blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            )

            # pressure arrays
            self.addArray(
                grid,
                "p",
                blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F"),
            )

            # velocity array
            array = np.column_stack(
                tuple(
                    [
                        blk.array["q"][ng:-ng, ng:-ng, ng:-ng, i].ravel(order="F")
                        for i in (1, 2, 3)
                    ]
                )
            )
            self.addArray(grid, "Velocity", array)

            # temperature arrays
            self.addArray(
                grid,
                "T",
                blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 4].ravel(order="F"),
            )

            for n, var in enumerate(blk.speciesNames[0:-1]):
                self.addArray(
                    grid,
                    var,
                    blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5 + n].ravel(order="F"),
                )

            # Add nth species
            array = numpy_support.numpy_to_vtk(
                1.0
                - np.sum(blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5::], axis=-1).ravel(
                    order="F"
                )
            )
            self.addArray(grid, blk.speciesNames[-1], array)

            mbds.SetBlock(blk.nblki, grid)

        self.dataDescription.GetInputDescriptionByName("input").SetGrid(mbds)

    def addArray(self, grid, arrayName, npArray):
        # convert incoming numpy array to vtk
        vtkArray = numpy_support.numpy_to_vtk(npArray)
        vtkArray.SetName(arrayName)
        grid.GetCellData().AddArray(vtkArray)

    def swapArray(self, grid, arrayName, npArray):
        grid.GetCellData().RemoveArray(arrayName)
        # convert incoming numpy array to vtk
        vtkArray = numpy_support.numpy_to_vtk(npArray)
        vtkArray.SetName(arrayName)
        grid.GetCellData().AddArray(vtkArray)

    def __call__(self, mb):

        self.dataDescription.SetTimeData(mb.tme, mb.nrt)

        if not self._coProcessor.RequestDataDescription(self.dataDescription):
            return

        mbds = self.dataDescription.GetInputDescriptionByName("input").GetGrid()
        for blk in mb:
            ng = blk.ng
            grid = mbds.GetBlock(blk.nblki)

            # density array
            self.swapArray(
                grid, "rho", blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            )

            # pressure array
            self.swapArray(
                grid, "p", blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            )

            # velocity array
            array = np.column_stack(
                tuple(
                    [
                        blk.array["q"][ng:-ng, ng:-ng, ng:-ng, i].ravel(order="F")
                        for i in (1, 2, 3)
                    ]
                )
            )
            self.addArray(grid, "Velocity", array)

            # temperature array
            self.swapArray(
                grid, "T", blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 4].ravel(order="F")
            )

            # species arrrays
            for n, var in enumerate(blk.speciesNames[0:-1]):
                self.swapArray(
                    grid,
                    var,
                    blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5 + n].ravel(order="F"),
                )

            # Add nth species
            array = 1.0 - np.sum(
                blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5::], axis=-1
            ).ravel(order="F")
            self.swapArray(grid, blk.speciesNames[-1], array.ravel(order="F"))

        # Execute coprocessing
        self._coProcessor.CoProcess(self.dataDescription)

    def finalize(self):
        self._coProcessor.Finalize()

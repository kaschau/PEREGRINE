import numpy as np

from paraview.simple import GetParaViewVersion
from paraview.modules import vtkPVCatalyst as catalyst
from paraview.modules import vtkPVPythonCatalyst as pythoncatalyst
import vtk
from paraview.vtk.util import numpy_support


class coprocessor:
    def __init__(self, mb, config):
        # Initialize
        self._coProcessor = catalyst.vtkCPProcessor()

        # Add the coproc script
        if str(GetParaViewVersion()) == "5.9":
            pipeline = pythoncatalyst.vtkCPPythonScriptV2Pipeline()
        elif str(GetParaViewVersion()) == "5.8":
            pipeline = pythoncatalyst.vtkCPPythonScriptPipeline()
        else:
            raise ValueError("Not a compatible paraview version")

        fileName = config["Catalyst"]["cpFile"]
        pipeline.Initialize(fileName)
        self._coProcessor.AddPipeline(pipeline)

        # Save the data descriptions
        self.dataDescription = catalyst.vtkCPDataDescription()
        # Add the input input
        self.dataDescription.AddInput("input")

        # Create the grid and data arrays
        mbds = vtk.vtkMultiBlockDataSet()
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
            rho = numpy_support.numpy_to_vtk(
                blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            )
            rho.SetName("rho")
            grid.GetCellData().AddArray(rho)

            # # pressure arrays
            # pressure = numpy_support.numpy_to_vtk(
            #     blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            # )
            # pressure.SetName("p")
            # grid.GetCellData().AddArray(pressure)

            # # velocity array
            # array = np.column_stack(
            #     tuple(
            #         [
            #             blk.array["q"][ng:-ng, ng:-ng, ng:-ng, i].ravel(order="F")
            #             for i in (1, 2, 3)
            #         ]
            #     )
            # )
            # velocity = numpy_support.numpy_to_vtk(array)
            # velocity.SetName("Velocity")
            # grid.GetCellData().AddArray(velocity)

            # # temperature arrays
            # temperature = numpy_support.numpy_to_vtk(
            #     blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 4].ravel(order="F")
            # )
            # temperature.SetName("T")
            # grid.GetCellData().AddArray(temperature)

            # for i, var in enumerate(blk.speciesNames[0:-1]):
            #     array = numpy_support.numpy_to_vtk(
            #         blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5 + i].ravel(order="F")
            #     )
            #     array.SetName(var)
            #     grid.GetCellData().AddArray(array)

            # # Add nth species
            # array = numpy_support.numpy_to_vtk(
            #     1.0
            #     - np.sum(blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5::], axis=-1).ravel(order="F")
            # )
            # array.SetName(blk.speciesNames[-1])
            # grid.GetCellData().AddArray(array)

            mbds.SetBlock(blk.nblki, grid)

        self.dataDescription.GetInputDescriptionByName("input").SetGrid(mbds)

    def __call__(self, mb):

        self.dataDescription.SetTimeData(mb.tme, mb.nrt)

        if not self._coProcessor.RequestDataDescription(self.dataDescription):
            return

        mbds = self.dataDescription.GetInputDescriptionByName("input").GetGrid()
        for blk in mb:
            ng = blk.ng
            grid = mbds.GetBlock(blk.nblki)
            ncls = grid.GetNumberOfCells()

            # density arrays
            rho = grid.GetCellData().GetArray("rho")
            for i, val in enumerate(
                blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            ):
                rho.SetValue(i, val)

            # array = numpy_support.numpy_to_vtk(
            #     blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            # )
            # rho.SetArray(
            #     array,
            #     ncls,
            #     1,
            # )

            # # pressure arrays
            # pressure = grid.GetCellData().GetArray("p")
            # pressure.SetArray(
            #     numpy_support.numpy_to_vtk(
            #         blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            #     ),
            #     ncls,
            #     1,
            # )

            # # velocity array
            # array = np.column_stack(
            #     tuple(
            #         [
            #             blk.array["q"][ng:-ng, ng:-ng, ng:-ng, i].ravel(order="F")
            #             for i in (1, 2, 3)
            #         ]
            #     )
            # )
            # Velocity = grid.GetCellData().GetArray("Velocity")
            # Velocity.SetArray(array, ncls * 3, 1)

            # # temperature arrays
            # temperature = grid.GetCellData().GetArray("T")
            # temperature.SetArray(
            #     numpy_support.numpy_to_vtk(
            #         blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 4].ravel(order="F")
            #     ),
            #     ncls,
            #     1,
            # )

            # for i, var in enumerate(blk.speciesNames[0:-1]):
            #     array = grid.GetCellData().GetArray(var)
            #     array.SetArray(
            #         numpy_support.numpy_to_vtk(
            #             blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5 + i].ravel(order="F")
            #         ),
            #         ncls,
            #         0,
            #     )

            # # Add nth species
            # array = grid.GetCellData().GetArray(blk.speciesNames[-1])
            # array.SetArray(
            #     numpy_support.numpy_to_vtk(
            #         1.0
            #         - np.sum(
            #             blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5::], axis=-1
            #         ).ravel(order="F")
            #     ),
            #     ncls,
            #     1,
            # )

        # Execute coprocessing
        self._coProcessor.CoProcess(self.dataDescription)

    def finalize(self):
        self._coProcessor.Finalize()

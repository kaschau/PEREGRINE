import numpy as np


class coprocessor:
    def __init__(self, mb, config):
        from paraview.simple import GetParaViewVersion
        from paraview.modules import vtkPVCatalyst as catalyst
        from paraview.modules import vtkPVPythonCatalyst as pythoncatalyst
        import vtk
        from paraview.vtk.util import numpy_support

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

            # pressure arrays
            pressure = numpy_support.numpy_to_vtk(
                blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            )
            pressure.SetName("p")
            grid.GetCellData().AddArray(pressure)

            # velocity array
            array = np.column_stack(
                tuple(
                    [
                        blk.array["q"][ng:-ng, ng:-ng, ng:-ng, i].ravel(order="F")
                        for i in (1, 2, 3)
                    ]
                )
            )
            velocity = numpy_support.numpy_to_vtk(array)
            velocity.SetName("Velocity")
            grid.GetCellData().AddArray(velocity)

            # temperature arrays
            temperature = numpy_support.numpy_to_vtk(
                blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 4].ravel(order="F")
            )
            temperature.SetName("T")
            grid.GetCellData().AddArray(temperature)

            for i, var in enumerate(blk.speciesNames[0:-1]):
                array = numpy_support.numpy_to_vtk(
                    blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5 + i].ravel(order="F")
                )
                array.SetName(var)
                grid.GetCellData().AddArray(array)

            # Add nth species
            array = numpy_support.numpy_to_vtk(
                1.0
                - np.sum(blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5::], axis=-1).ravel(
                    order="F"
                )
            )
            array.SetName(blk.speciesNames[-1])
            grid.GetCellData().AddArray(array)

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

            # density array
            rho = grid.GetCellData().GetArray("rho")
            for i, val in enumerate(
                blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            ):
                rho.SetValue(i, val)

            # pressure array
            p = grid.GetCellData().GetArray("p")
            for i, val in enumerate(
                blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            ):
                p.SetValue(i, val)

            # velocity array
            V = grid.GetCellData().GetArray("Velocity")
            shape = (blk.ni - 1, blk.nj - 1, blk.nk - 1, 3)
            for i, val in enumerate(
                blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 1:4]
                .reshape(-1, shape[-1], order="F")
                .ravel()
            ):
                V.SetValue(i, val)

            # temperature array
            T = grid.GetCellData().GetArray("T")
            for i, val in enumerate(
                blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 4].ravel(order="F")
            ):
                T.SetValue(i, val)

            # species arrrays
            for n, var in enumerate(blk.speciesNames[0:-1]):
                N = grid.GetCellData().GetArray(var)
                for i, val in enumerate(
                    blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5 + n].ravel(order="F")
                ):
                    N.SetValue(i, val)

            # Add nth species
            N = grid.GetCellData().GetArray(blk.speciesNames[-1])
            for i, val in enumerate(
                1.0
                - np.sum(blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5::], axis=-1).ravel(
                    order="F"
                )
            ):
                N.SetValue(i, val)

        # Execute coprocessing
        self._coProcessor.CoProcess(self.dataDescription)

    def finalize(self):
        self._coProcessor.Finalize()

import sys

# import paraview.simple
from paraview import servermanager
from paraview.modules import vtkPVCatalyst as catalyst
from paraview.modules import vtkPVPythonCatalyst as pythoncatalyst
import vtk
from paraview.vtk.util import numpy_support


class coprocessor:
    def __init__(self, mb, config):
        # Initialize
        if not servermanager.vtkProcessModule.GetProcessModule():
            servermanager.vtkInitializationHelper.Initialize(
                sys.executable, servermanager.vtkProcessModule.PROCESS_BATCH, None
            )
        self._coProcessor = catalyst.vtkCPProcessor()

        # Add the coproc script
        pipeline = pythoncatalyst.vtkCPPythonScriptPipeline()
        fileName = config["coproc"]["cpFile"]
        pipeline.Initialize(fileName)
        self._coProcessor.AddPipeline(pipeline)

        # Save the data descriptions
        self.dataDescription = catalyst.vtkCPDataDescription()
        # Set the time and time step
        self.dataDescription.SetTimeData(mb.tme, mb.nrt)
        # Add the input input
        self.dataDescription.AddInput("input")

        # Create the grid and data arrays
        mbds = vtk.vtkMultiBlockDataSet()

    def finalize(self):
        self._coProcessor.Finalize()
        # if we are running through Python we need to finalize extra stuff
        # to avoid memory leak messages.
        import sys
        import ntpath

        if ntpath.basename(sys.executable) == "python":
            servermanager.vtkInitializationHelper.Finalize()

    def __call__(self, mb):

        self.dataDescription.SetTimeData(mb.nrt, mb.tme)

        if self._coProcessor.RequestDataDescription(self.dataDescription):

            imageData = vtk.vtkImageData()
            imageData.SetExtent(
                grid.XStartPoint,
                grid.XEndPoint,
                0,
                grid.NumberOfYPoints - 1,
                0,
                grid.NumberOfZPoints - 1,
            )
            imageData.SetSpacing(grid.Spacing)

            velocity = numpy_support.numpy_to_vtk(attributes.Velocity)
            velocity.SetName("velocity")
            imageData.GetPointData().AddArray(velocity)

            pressure = numpy_support.numpy_to_vtk(attributes.Pressure)
            pressure.SetName("pressure")
            imageData.GetCellData().AddArray(pressure)
            dataDescription.GetInputDescriptionByName("input").SetGrid(imageData)
            dataDescription.GetInputDescriptionByName("input").SetWholeExtent(
                0,
                grid.NumberOfGlobalXPoints - 1,
                0,
                grid.NumberOfYPoints - 1,
                0,
                grid.NumberOfZPoints - 1,
            )
            self._coProcessor.CoProcess(self.dataDescription)

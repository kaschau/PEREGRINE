# PEREGRINE Documentation


## Executable Mode

PEREGRINE can run in both a scriptable mode for simple cases, or executable mode for larger production runs. For examples of scripting modes, see the [examples](https://github.com/kaschau/PEREGRINE/tree/main/examples) directory in the repository. For executable mode, see the case directory structure [here](./executableMode.md), as well as the format the the input config file below.

## CoProcessing

Coprocessing with ParaView is amazing, but takes effort to make it work well in my experience. It seems they have cleaned it up a lot from back in the day. To start download and install Paraview from source. I have tested up to 5.11 and it works well for me. To make coprocessing work, you need to compile paraview in catalyst mode. With cmake, pass the argument:

> ccmake -DPARAVIEW_BUILD_EDITION:STRING=CATALYST /path/to/ParaView_src/

Assume you are installing it in `/path/to/paraview`.

### Some Tips and Tricks

You must set the cmake flags:
```
PARAVIEW_USE_MPI=ON
PARAVIEW_USE_PYTHON=ON
```

Make sure ParaView finds the correct python you are planning to use with PEREGRINE.

Once you configure a bunch of times, look for a group of options that look like `VTK_MODULE_USE_EXTERNAL_VTK_*`. You want to turn on as many of those as you can, otherwise ParaView will download, and fail, to install many of these. Especially `mpi4py`, `png`, `libxml` and so on. Hopefully your cluster has these already and you can module load them, or your package manager probably has them too.


### Running

To get it to work, you have to set these environment variables so python can find the catalyst install.

```
export PYTHONPATH=$PYTHONPATH:/path/to/paraview/lib/python3.XX/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/paraview/lib
export PYTHONPATH=$PYTHONPATH:/path/to/paraview/lib
```

You will know it worked if `from paraview.catalyst import bridge` works in an interpreter.

Good Luck!


## Code Structure
  * [Python + Kokkos](./codeStructure/pythonKokkos.md)
  * [Multiblock Structure](./codeStructure/multiblock.md)
  * [Array, Mirror, View](./codeStructure/arrayMirrorView.md)
  * [Boundary Conditions](./boundaryConditions.md)

## Utilities
There are a bunch of utilities to make running with PEREGRINE easier, see the [utilities](https://github.com/kaschau/PEREGRINE/tree/main/utilities) folder. They are pretty obvious from the name. Here is a brief list and summary.

 * [ct2pgChem](https://github.com/kaschau/PEREGRINE/blob/main/utilities/ct2pgChem.py): For performance, PEREGRINE hand writes the chemical source term kernel for a given chemistry mechanism. This utility takes a cantera yaml file and outputs a C++ source code ready to be used in PEREGRINE. You just need to add the C++ code to the appropriate folder in `src` (make sure to add the pybind11 bindings as well), and the `compute` module will have access to your new chemical source terms.

 * [ct2pgTHTR](https://github.com/kaschau/PEREGRINE/blob/main/utilities/ct2pgTHTR.py): Thermodynamic and transport properties are given to PEREGRINE via a stripped down, custom yaml file that is read in at run time. This utility creates that file from a cantera yaml file.

 * [cutGrid](https://github.com/kaschau/PEREGRINE/blob/main/utilities/cutGrid.py): This utility decomposes a PEREGRINE grid into smaller blocks be performing persistent number cuts of blocks along a specified axis. YAY MULTIBLOCK!

 * [generateTracePoints](https://github.com/kaschau/PEREGRINE/blob/main/utilities/generateTracePoints.py): Generate trace point input file for collecting data at points during a simulation.

 * [gridPro2pg](https://github.com/kaschau/PEREGRINE/blob/main/utilities/gridPro2pg.py): Grid Pro mesh to PEREGRINE mesh translation.

 * [icem2pg](https://github.com/kaschau/PEREGRINE/blob/main/utilities/icem2pg.py): ICEM mesh to PEREGRINE mesh translation.

 * [interpolate](https://github.com/kaschau/PEREGRINE/blob/main/utilities/interpolate.py): Interpolate results from one mesh to another mesh.

 * [loadBalancer](https://github.com/kaschau/PEREGRINE/blob/main/utilities/loadBalancer.py): Distribute blocks amongst a set number of groups.

 * [verifyGrid](https://github.com/kaschau/PEREGRINE/blob/main/utilities/verifyGrid.py): Go block by block, face by face, and make sure everyone agrees who is connected to who, and in what orientation.

## Config File

PEREGRINE run in executable mode requires an input configuration file (in yaml format) to tell it what to do. Here is a sample of such a file with comments.

    # input output directories
    io:
     gridDir: ./Grid
     inputDir: ./Input
     restartDir: ./Restart
     archiveDir: ./Archive

    # simulation control
    simulation:
      niter: 10
      dt: 0.9
      restartFrom: 0
      animateRestart: true
      animateArchive: true
      niterRestart: 1 #save out restart files
      niterArchive: 10 #save out light weight single precision files
      niterPrint: 1  #print to stdout case progress
      variableTimeStep: true
      maxCFL: 1.0
      checkNan: 1

    solver: # solver selection
      timeIntegration: rk1

    RHS: # RHS control
      shockHandling: null #method of shock handling
      primaryAdvFlux: secondOrderKEEP  # how to compute low dissipative adctive flux
      secondaryAdvFlux: null # how to compute low order stable advective flux
      switchAdvFlux: null # how to switch between the two
      diffusion: true #solve diffusion terms?
      subgrid: null #use a subgrid model?

    thermochem: #thermo, transport, chem options
      spdata: ["O2", "N2"] # list of species names, or path to THTR input yaml
      eos: cpg
      trans: constantProps #transport property calculations
      chemistry: false
      mechanism: null #name of mechanism compiled in the compute module.

    coprocess: # coprocessing
      catalyst: false # or path to paraview catalyst input python file
      trace: false # or path to trace points file
      niterTrace: 1
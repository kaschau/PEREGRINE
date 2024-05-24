# PEREGRINE Documentation


## Executable Mode

PEREGRINE can run in both a scriptable mode for simple cases, or executable mode for larger production runs. For examples of scripting modes, see [examples](../examples). For executable mode, see the case directory structure [here](./executableMode.md).

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

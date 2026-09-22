# PEREGRINE documentation

- [Installing](install.md): a Kokkos install, the Python dependencies, the
  kernel cache, MPI and parallel HDF5, profiling.
- [Running a case](running.md): a case in a script, executable mode with
  the `peregrine` command, starting from a state you make, the tools, the
  examples.
- [The config file](config.md): every section and key, what it does, and
  what is refused.
- [The mixture](mixture.md): where species come from, the equation of
  state, transport and diffusion models, what each needs, the fits, chemistry.
- [Boundary conditions](boundaryConditions.md): how a face gets a name,
  what each condition reads, profiles.
- [Plugins](plugins.md): reporting, writing results, checking for NaNs,
  tracing points, the viscous sponge, Catalyst.
- [Files](files.md): the grid, partitions, results, the config file, the
  small input files.
- [Code structure](codeStructure.md): the packages by role, the compute
  tree, how a kernel is compiled, the tests.

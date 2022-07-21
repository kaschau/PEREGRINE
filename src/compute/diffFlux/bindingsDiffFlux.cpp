#include "compute.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindDiffFlux(py::module_ &m) {
  // ./diffFlux
  py::module diffFlux = m.def_submodule("diffFlux", "diffusive flux module");
  //  |----> diffusiveFlux.cpp
  diffFlux.def("diffusiveFlux", &diffusiveFlux,
               "Compute centeral difference viscous fluxes. Order set by dqdx",
               py::arg("block_ object"));
}

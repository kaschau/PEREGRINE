#include "compute.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindBoundaryConditions(py::module_ &m) {
  // ./boundaryConditions
  py::module bcs = m.def_submodule("bcs", "boundary conditions module");
  //  |----> inlets.cpp
  py::module inlets = bcs.def_submodule("inlets", "inlet boundary conditions module");
  inlets.def("constantVelocitySubsonicInlet", &constantVelocitySubsonicInlet, "Const velo subsonic inlet",
        py::arg("block_ object"),
        py::arg("face_ object"));
}

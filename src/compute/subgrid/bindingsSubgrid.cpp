#include "subgrid.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindSubgrid(py::module_ &m) {
  // ./subgrid
  py::module subgrid = m.def_submodule("subgrid", "subgrid");
  subgrid.def("smagorinsky", &smagorinsky,
              "Compute subgrid viscosity from Smagorisnsky model.",
              py::arg("block_ object"));
}

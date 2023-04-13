#include "subgrid.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bindSubgrid(nb::module_ &m) {
  // ./subgrid
  nb::module_ subgrid = m.def_submodule("subgrid", "subgrid");
  //  |----> mixedScaleModel.cpp
  subgrid.def("mixedScaleModel", &mixedScaleModel,
              "Compute subgrid viscosity from mixed scale model.",
              nb::arg("block_ object"));
  subgrid.def("smagorinsky", &smagorinsky,
              "Compute subgrid viscosity from Smagorisnsky model.",
              nb::arg("block_ object"));
}

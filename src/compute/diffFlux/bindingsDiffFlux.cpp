#include "diffFlux.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bindDiffFlux(nb::module_ &m) {
  // ./diffFlux
  nb::module_ diffFlux = m.def_submodule("diffFlux", "diffusive flux module");
  //  |----> diffusiveFlux.cpp
  diffFlux.def("diffusiveFlux", &diffusiveFlux,
               "Compute centeral difference viscous fluxes. Order set by dqdx",
               nb::arg("block_ object"));
  //  |----> alphaDamping.cpp
  diffFlux.def("alphaDampingFlux", &alphaDampingFlux,
               "Evaluate face derivatices with alpha damping.",
               nb::arg("block_ object"));
}

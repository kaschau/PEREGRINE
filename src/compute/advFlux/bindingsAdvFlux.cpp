#include "advFlux.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bindAdvFlux(nb::module_ &m) {
  // ./advFlux
  nb::module_ advFlux = m.def_submodule("advFlux", "advective flux module");
  //  |----> secondOrderKEEP.cpp
  advFlux.def("secondOrderKEEP", &secondOrderKEEP,
              "Compute centeral difference euler fluxes via second order KEEP",
              nb::arg("block_ object"));
  //  |----> centeredDifference.cpp
  advFlux.def("centralDifference", &centralDifference,
              "Compute central difference euler fluxes",
              nb::arg("block_ object"));
  //  |----> scalarDissipation.cpp
  advFlux.def("scalarDissipation", &scalarDissipation,
              "Compute scalar dissipation", nb::arg("block_ object"));
  //  |----> fourthOrderKEEP.cpp
  advFlux.def("fourthOrderKEEP", &fourthOrderKEEP,
              "Compute centeral difference euler fluxes via fourth order KEEP",
              nb::arg("block_ object"));
  //  |----> rusanov.cpp
  advFlux.def("rusanov", &rusanov,
              "Compute first order euler fluxes via rusanov",
              nb::arg("block_ object"));
  //  |----> ausmPlusUp.cpp
  advFlux.def("ausmPlusUp", &ausmPlusUp, "Compute inviscid fluxes via AUSM+UP",
              nb::arg("block_ object"));
  //  |----> hllc.cpp
  advFlux.def("hllc", &hllc, "Compute first order euler fluxes via hllc",
              nb::arg("block_ object"));
  //  |----> muscl2hllc.cpp
  advFlux.def(
      "muscl2hllc", &muscl2hllc,
      "Compute first order euler fluxes via 2nd order MUSCL with hllc flux",
      nb::arg("block_ object"));
  //  |----> muscl2rusanov.cpp
  advFlux.def(
      "muscl2rusanov", &muscl2rusanov,
      "Compute first order euler fluxes via 2nd order MUSCL with rusanov flux",
      nb::arg("block_ object"));
}

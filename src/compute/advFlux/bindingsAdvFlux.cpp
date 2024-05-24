#include "advFlux.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindAdvFlux(py::module_ &m) {
  // ./advFlux
  py::module advFlux = m.def_submodule("advFlux", "advective flux module");
  //  |----> KEEP.cpp
  advFlux.def("KEEP", &KEEP,
              "Compute centeral difference euler fluxes via second order KEEP",
              py::arg("block_ object"));
  //  |----> KEEPpe.cpp
  advFlux.def(
      "KEEPpe", &KEEPpe,
      "Compute centeral difference euler fluxes via second order KEEPpe",
      py::arg("block_ object"));
  //  |----> KEPaEC.cpp
  advFlux.def(
      "KEPaEC", &KEPaEC,
      "Compute centeral difference euler fluxes via second order KEPaEC",
      py::arg("block_ object"));
  //  |----> centeredDifference.cpp
  advFlux.def("centralDifference", &centralDifference,
              "Compute central difference euler fluxes",
              py::arg("block_ object"));
  //  |----> scalarDissipation.cpp
  advFlux.def("scalarDissipation", &scalarDissipation,
              "Compute scalar dissipation", py::arg("block_ object"));
  //  |----> fourthOrderKEEP.cpp
  advFlux.def("fourthOrderKEEP", &fourthOrderKEEP,
              "Compute centeral difference euler fluxes via fourth order KEEP",
              py::arg("block_ object"));
  //  |----> rusanov.cpp
  advFlux.def("rusanov", &rusanov,
              "Compute first order euler fluxes via rusanov",
              py::arg("block_ object"));
  //  |----> ausmPlusUp.cpp
  advFlux.def("ausmPlusUp", &ausmPlusUp, "Compute inviscid fluxes via AUSM+UP",
              py::arg("block_ object"));
  //  |----> hllc.cpp
  advFlux.def("hllc", &hllc, "Compute first order euler fluxes via hllc",
              py::arg("block_ object"));
  //  |----> muscl2hllc.cpp
  advFlux.def(
      "muscl2hllc", &muscl2hllc,
      "Compute first order euler fluxes via 2nd order MUSCL with hllc flux",
      py::arg("block_ object"));
  //  |----> muscl2rusanov.cpp
  advFlux.def(
      "muscl2rusanov", &muscl2rusanov,
      "Compute first order euler fluxes via 2nd order MUSCL with rusanov flux",
      py::arg("block_ object"));
}

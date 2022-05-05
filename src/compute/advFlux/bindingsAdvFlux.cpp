#include "compute.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindAdvFlux(py::module_ &m) {
  // ./advFlux
  py::module advFlux = m.def_submodule("advFlux", "advective flux module");
  //  |----> secondOrderKEEP.cpp
  advFlux.def("secondOrderKEEP", &secondOrderKEEP, "Compute centeral difference euler fluxes via second order KEEP",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));
  //  |----> jamesonDissipation.cpp
  advFlux.def("jamesonDissipation", &jamesonDissipation, "Compute jameson dissipation",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));
  //  |----> fourthOrderKEEP.cpp
  advFlux.def("fourthOrderKEEP", &fourthOrderKEEP, "Compute centeral difference euler fluxes via fourth order KEEP",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));
  //  |----> rusanov.cpp
  advFlux.def("rusanov", &rusanov, "Compute first order euler fluxes via rusanov",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));
  //  |----> ausmPlusUp.cpp
  advFlux.def("ausmPlusUp", &ausmPlusUp, "Compute inviscid fluxes via AUSM+UP",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));
  //  |----> hllc.cpp
  advFlux.def("hllc", &hllc, "Compute first order euler fluxes via hllc",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));
  //  |----> muscl2hllc.cpp
  advFlux.def("muscl2hllc", &muscl2hllc, "Compute first order euler fluxes via 2nd order MUSCL with hllc flux",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));
}

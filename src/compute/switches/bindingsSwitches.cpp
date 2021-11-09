#include "compute.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindSwitches(py::module_ &m) {
  // ./switches
  py::module switches = m.def_submodule("switches", "switches");
  //  |----> jameson.cpp
  switches.def("jamesonEntropy", &jamesonEntropy, "Compute switches based on entropy",
        py::arg("block_ object"));
  switches.def("jamesonPressure", &jamesonPressure, "Compute switches based on pressure",
        py::arg("block_ object"));
  //  |----> vanAlbada.cpp
  switches.def("vanAlbadaEntropy", &vanAlbadaEntropy, "Compute switches based on van Albada limiter",
        py::arg("block_ object"));
  switches.def("vanAlbadaPressure", &vanAlbadaPressure, "Compute switches based on van Albada limiter",
        py::arg("block_ object"));
}

#include "switches.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindSwitches(py::module_ &m) {
  // ./switches
  py::module switches = m.def_submodule("switches", "switches");
  switches.def("jamesonPressure", &jamesonPressure,
               "Compute switches based on pressure", py::arg("block_ object"));
  switches.def("vanAlbadaPressure", &vanAlbadaPressure,
               "Compute switches based on van Albada limiter",
               py::arg("block_ object"));
  //  |----> vanLeer.cpp
  switches.def("vanLeer", &vanLeer,
               "Compute switches based on modified van Leer limiter",
               py::arg("block_ object"));
}

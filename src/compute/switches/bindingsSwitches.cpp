#include "switches.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bindSwitches(nb::module_ &m) {
  // ./switches
  nb::module_ switches = m.def_submodule("switches", "switches");
  switches.def("jamesonPressure", &jamesonPressure,
               "Compute switches based on pressure", nb::arg("block_ object"));
  switches.def("vanAlbadaPressure", &vanAlbadaPressure,
               "Compute switches based on van Albada limiter",
               nb::arg("block_ object"));
  //  |----> vanLeer.cpp
  switches.def("vanLeer", &vanLeer,
               "Compute switches based on modified van Leer limiter",
               nb::arg("block_ object"));
}

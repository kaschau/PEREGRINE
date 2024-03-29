#include "thermo.hpp"
#include "thtrdat_.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindThermo(py::module_ &m) {
  // ./thermo
  py::module thermo = m.def_submodule("thermo", "thermo module");
  //  |----> cpg.cpp
  thermo.def(
      "cpg", &cpg, "Update primatives or conservatives with cpg assumption",
      py::arg("block_ object"), py::arg("thtrdat_ object"), py::arg("nface"),
      py::arg("given"), py::arg("i") = 0, py::arg("j") = 0, py::arg("k") = 0);
  //  |----> tpg.cpp
  thermo.def(
      "tpg", &tpg, "Update primatives or conservatives with tpg assumption",
      py::arg("block_ object"), py::arg("thtrdat_ object"), py::arg("nface"),
      py::arg("given"), py::arg("i") = 0, py::arg("j") = 0, py::arg("k") = 0);
  //  |----> cubic.cpp
  thermo.def(
      "cubic", &cubic, "Update primatives or conservatives with cubic EOS",
      py::arg("block_ object"), py::arg("thtrdat_ object"), py::arg("nface"),
      py::arg("given"), py::arg("i") = 0, py::arg("j") = 0, py::arg("k") = 0);

  ////////////////////////////////////////////////////////////////////////////////
  ///////////////////  C++ Parent thtrdat_ class
  ////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  py::class_<thtrdat_>(thermo, "thtrdat_", py::dynamic_attr())
      .def(py::init<>())

      .def_readwrite("ns", &thtrdat_::ns)
      .def_readonly("Ru", &thtrdat_::Ru)

      .def_readwrite("MW", &thtrdat_::MW)

      .def_readwrite("NASA7", &thtrdat_::NASA7)

      .def_readwrite("muPoly", &thtrdat_::muPoly)
      .def_readwrite("kappaPoly", &thtrdat_::kappaPoly)
      .def_readwrite("DijPoly", &thtrdat_::DijPoly)

      .def_readwrite("cp0", &thtrdat_::cp0)
      .def_readwrite("mu0", &thtrdat_::mu0)
      .def_readwrite("kappa0", &thtrdat_::kappa0)

      .def_readwrite("Tcrit", &thtrdat_::Tcrit)
      .def_readwrite("pcrit", &thtrdat_::pcrit)
      .def_readwrite("Vcrit", &thtrdat_::Vcrit)
      .def_readwrite("acentric", &thtrdat_::acentric)

      .def_readwrite("chungA", &thtrdat_::chungA)
      .def_readwrite("chungB", &thtrdat_::chungB)
      .def_readwrite("redDipole", &thtrdat_::redDipole);
}

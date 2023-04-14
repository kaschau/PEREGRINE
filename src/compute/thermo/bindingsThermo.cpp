#include "thermo.hpp"
#include "thtrdat_.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

void bindThermo(nb::module_ &m) {
  // ./thermo
  nb::module_ thermo = m.def_submodule("thermo", "thermo module");
  //  |----> cpg.cpp
  thermo.def(
      "cpg", &cpg, "Update primatives or conservatives with cpg assumption",
      nb::arg("block_ object"), nb::arg("thtrdat_ object"), nb::arg("nface"),
      nb::arg("given"), nb::arg("i") = 0, nb::arg("j") = 0, nb::arg("k") = 0);
  //  |----> tpg.cpp
  thermo.def(
      "tpg", &tpg, "Update primatives or conservatives with tpg assumption",
      nb::arg("block_ object"), nb::arg("thtrdat_ object"), nb::arg("nface"),
      nb::arg("given"), nb::arg("i") = 0, nb::arg("j") = 0, nb::arg("k") = 0);
  //  |----> cubic.cpp
  thermo.def(
      "cubic", &cubic, "Update primatives or conservatives with cubic EOS",
      nb::arg("block_ object"), nb::arg("thtrdat_ object"), nb::arg("nface"),
      nb::arg("given"), nb::arg("i") = 0, nb::arg("j") = 0, nb::arg("k") = 0);

  ////////////////////////////////////////////////////////////////////////////////
  ///////////////////  C++ Parent thtrdat_ class
  ////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  nb::class_<thtrdat_>(thermo, "thtrdat_", nb::dynamic_attr())
      .def(nb::init<>())

      .def_rw("ns", &thtrdat_::ns)
      .def_ro("Ru", &thtrdat_::Ru)

      .def_rw("MW", &thtrdat_::MW)

      .def_rw("NASA7", &thtrdat_::NASA7)

      .def_rw("muPoly", &thtrdat_::muPoly)
      .def_rw("kappaPoly", &thtrdat_::kappaPoly)
      .def_rw("DijPoly", &thtrdat_::DijPoly)

      .def_rw("cp0", &thtrdat_::cp0)
      .def_rw("mu0", &thtrdat_::mu0)
      .def_rw("kappa0", &thtrdat_::kappa0)

      .def_rw("Tcrit", &thtrdat_::Tcrit)
      .def_rw("pcrit", &thtrdat_::pcrit)
      .def_rw("Vcrit", &thtrdat_::Vcrit)
      .def_rw("acentric", &thtrdat_::acentric)

      .def_rw("chungA", &thtrdat_::chungA)
      .def_rw("chungB", &thtrdat_::chungB)
      .def_rw("redDipole", &thtrdat_::redDipole);
}

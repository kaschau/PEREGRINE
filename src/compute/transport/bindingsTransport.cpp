#include "transport.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bindTransport(nb::module_ &m) {
  // ./transport
  nb::module_ transport = m.def_submodule("transport", "transport module");
  //  |----> kineticTheory.cpp
  transport.def(
      "kineticTheory", &kineticTheory,
      "Update transport properties from primatives via kinetic theory",
      nb::arg("block_"), nb::arg("thtrdat_ object"), nb::arg("face"),
      nb::arg("i") = 0, nb::arg("j") = 0, nb::arg("k") = 0);
  //  |----> constantProps.cpp
  transport.def(
      "constantProps", &constantProps,
      "Update transport properties from primatives with constant properties",
      nb::arg("block_"), nb::arg("thtrdat_ object"), nb::arg("face"),
      nb::arg("i") = 0, nb::arg("j") = 0, nb::arg("k") = 0);
  //  |----> kineticTheoryUnityLewis.cpp
  transport.def("kineticTheoryUnityLewis", &kineticTheoryUnityLewis,
                "Update transport properties from primatives via kinetic "
                "theory, using unity Lewis assumption for Dij",
                nb::arg("block_"), nb::arg("thtrdat_ object"), nb::arg("face"),
                nb::arg("i") = 0, nb::arg("j") = 0, nb::arg("k") = 0);
  //  |----> chungDenseGasUnityLewis.cpp
  transport.def("chungDenseGasUnityLewis", &chungDenseGasUnityLewis,
                "Update transport properties from primatives via Chung's "
                "correleation for dense fluid and unity Lewis number approx.",
                nb::arg("block_"), nb::arg("thtrdat_ object"), nb::arg("face"),
                nb::arg("i") = 0, nb::arg("j") = 0, nb::arg("k") = 0);
}

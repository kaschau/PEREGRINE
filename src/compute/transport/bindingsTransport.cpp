#include "compute.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindTransport(py::module_ &m) {
  // ./transport
  py::module transport = m.def_submodule("transport", "transport module");
  //  |----> kineticTheory.cpp
  transport.def(
      "kineticTheory", &kineticTheory,
      "Update transport properties from primatives via kinetic theory",
      py::arg("block_"), py::arg("thtrdat_ object"), py::arg("face"),
      py::arg("i") = 0, py::arg("j") = 0, py::arg("k") = 0);
  //  |----> constantProps.cpp
  transport.def(
      "constantProps", &constantProps,
      "Update transport properties from primatives with constant properties",
      py::arg("block_"), py::arg("thtrdat_ object"), py::arg("face"),
      py::arg("i") = 0, py::arg("j") = 0, py::arg("k") = 0);
  //  |----> kineticTheoryUnityLewis.cpp
  transport.def("kineticTheoryUnityLewis", &kineticTheoryUnityLewis,
                "Update transport properties from primatives via kinetic "
                "theory, using unity Lewis assumption for Dij",
                py::arg("block_"), py::arg("thtrdat_ object"), py::arg("face"),
                py::arg("i") = 0, py::arg("j") = 0, py::arg("k") = 0);
  //  |----> chungDenseGasUnityLewis.cpp
  transport.def("chungDenseGasUnityLewis", &chungDenseGasUnityLewis,
                "Update transport properties from primatives via Chung's "
                "correleation for dense fluid and unity Lewis number approx.",
                py::arg("block_"), py::arg("thtrdat_ object"), py::arg("face"),
                py::arg("i") = 0, py::arg("j") = 0, py::arg("k") = 0);
}

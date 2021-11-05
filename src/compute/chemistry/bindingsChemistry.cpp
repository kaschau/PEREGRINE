#include "compute.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindChemistry(py::module_ &m) {
  // ./chemistry
  py::module chemistry = m.def_submodule("chemistry", "chemistry module");
  //  |----> CH4_O2_Stanford_Skeletal.cpp
  chemistry.def("chem_CH4_O2_Stanford_Skeletal", &chem_CH4_O2_Stanford_Skeletal, "Chemical source terms from",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"),
        py::arg("face")=0,
        py::arg("i")=0,
        py::arg("j")=0,
        py::arg("k")=0);
  //  |----> GRI30
  chemistry.def("chem_GRI30", &chem_GRI30, "Chemical source terms from GRI3.0",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"),
        py::arg("face")=0,
        py::arg("i") = 0,
        py::arg("j") = 0,
        py::arg("k") = 0);
}

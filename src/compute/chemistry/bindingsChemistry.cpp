#include "chemistry.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindChemistry(py::module_ &m) {
  // ./chemistry
  py::module chemistry = m.def_submodule("chemistry", "chemistry module");
  //  |----> C2H4_Air_Red22.cpp
  chemistry.def("chem_C2H4_Air_Red22", &chem_C2H4_Air_Red22,
                "Chemical source terms from", py::arg("block_ object"),
                py::arg("thtrdat_ object"), py::arg("face") = 0,
                py::arg("i") = 0, py::arg("j") = 0, py::arg("k") = 0,
                py::arg("nChemSubSteps") = 1, py::arg("dt") = 1.0);
  //  |----> C2H4_Air_Skeletal.cpp
  chemistry.def("chem_C2H4_Air_Skeletal", &chem_C2H4_Air_Skeletal,
                "Chemical source terms from", py::arg("block_ object"),
                py::arg("thtrdat_ object"), py::arg("face") = 0,
                py::arg("i") = 0, py::arg("j") = 0, py::arg("k") = 0,
                py::arg("nChemSubSteps") = 1, py::arg("dt") = 1.0);
  //  |----> CH4_O2_FFCMY.cpp
  chemistry.def("chem_CH4_O2_FFCMY", &chem_CH4_O2_FFCMY,
                "Chemical source terms from", py::arg("block_ object"),
                py::arg("thtrdat_ object"), py::arg("face") = 0,
                py::arg("i") = 0, py::arg("j") = 0, py::arg("k") = 0,
                py::arg("nChemSubSteps") = 1, py::arg("dt") = 1.0);
  //  |----> GRI30
  chemistry.def("chem_GRI30", &chem_GRI30, "Chemical source terms from GRI3.0",
                py::arg("block_ object"), py::arg("thtrdat_ object"),
                py::arg("face") = 0, py::arg("i") = 0, py::arg("j") = 0,
                py::arg("k") = 0, py::arg("nChemSubSteps") = 1,
                py::arg("dt") = 1.0);
}

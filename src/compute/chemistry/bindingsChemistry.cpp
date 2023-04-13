#include "chemistry.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bindChemistry(nb::module_ &m) {
  // ./chemistry
  nb::module_ chemistry = m.def_submodule("chemistry", "chemistry module");
  //  |----> C2H4_Air_Red22.cpp
  chemistry.def("chem_C2H4_Air_Red22", &chem_C2H4_Air_Red22,
                "Chemical source terms from", nb::arg("block_ object"),
                nb::arg("thtrdat_ object"), nb::arg("face") = 0,
                nb::arg("i") = 0, nb::arg("j") = 0, nb::arg("k") = 0,
                nb::arg("nChemSubSteps") = 1, nb::arg("dt") = 1.0);
  //  |----> C2H4_Air_Skeletal.cpp
  chemistry.def("chem_C2H4_Air_Skeletal", &chem_C2H4_Air_Skeletal,
                "Chemical source terms from", nb::arg("block_ object"),
                nb::arg("thtrdat_ object"), nb::arg("face") = 0,
                nb::arg("i") = 0, nb::arg("j") = 0, nb::arg("k") = 0,
                nb::arg("nChemSubSteps") = 1, nb::arg("dt") = 1.0);
  //  |----> CH4_O2_Stanford_Skeletal.cpp
  chemistry.def("chem_CH4_O2_Stanford_Skeletal", &chem_CH4_O2_Stanford_Skeletal,
                "Chemical source terms from", nb::arg("block_ object"),
                nb::arg("thtrdat_ object"), nb::arg("face") = 0,
                nb::arg("i") = 0, nb::arg("j") = 0, nb::arg("k") = 0,
                nb::arg("nChemSubSteps") = 1, nb::arg("dt") = 1.0);
  //  |----> GRI30
  chemistry.def("chem_GRI30", &chem_GRI30, "Chemical source terms from GRI3.0",
                nb::arg("block_ object"), nb::arg("thtrdat_ object"),
                nb::arg("face") = 0, nb::arg("i") = 0, nb::arg("j") = 0,
                nb::arg("k") = 0, nb::arg("nChemSubSteps") = 1,
                nb::arg("dt") = 1.0);
}

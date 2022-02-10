#include "compute.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

void bindBoundaryConditions(py::module_ &m) {
  // ./boundaryConditions
  py::module bcs = m.def_submodule("bcs", "boundary conditions module");

  //  |----> inlets.cpp
  py::module inlets = bcs.def_submodule("inlets", "inlet boundary conditions module");
  inlets.def("constantVelocitySubsonicInlet", &constantVelocitySubsonicInlet, "Const velo subsonic inlet",
             py::arg("block_ object"),
             py::arg("face_ object"),
             py::arg("eos pointer"),
             py::arg("thtrdat_ object"),
             py::arg("terms"));
  inlets.def("supersonicInlet", &supersonicInlet, "Supersonic inlet",
             py::arg("block_ object"),
             py::arg("face_ object"),
             py::arg("eos pointer"),
             py::arg("thtrdat_ object"),
             py::arg("terms"));
  inlets.def("constantMassFluxSubsonicInlet", &constantMassFluxSubsonicInlet, "Const mass flux subsonic inlet",
             py::arg("block_ object"),
             py::arg("face_ object"),
             py::arg("eos pointer"),
             py::arg("thtrdat_ object"),
             py::arg("terms"));

  //  |----> walls.cpp
  py::module walls = bcs.def_submodule("walls", "walls boundary conditions module");
  walls.def("adiabaticNoSlipWall", &adiabaticNoSlipWall, "Adiabatic no slip wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"));
  walls.def("adiabaticSlipWall", &adiabaticSlipWall, "Adiabatic slip wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"));
  walls.def("adiabaticMovingWall", &adiabaticMovingWall, "Adiabatic moving wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"));
  walls.def("isoTNoSlipWall", &isoTNoSlipWall, "IsoT no slip wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"));
  walls.def("isoTSlipWall", &isoTSlipWall, "IsoT slip wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"));
  walls.def("isoTMovingWall", &isoTMovingWall, "Iso thermal moving wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"));

  //  |----> exits.cpp
  py::module exits = bcs.def_submodule("exits", "exit boundary conditions module");
  exits.def("constantPressureSubsonicExit", &constantPressureSubsonicExit, "Const pressure subsonic exit",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"));
  exits.def("supersonicExit", &supersonicExit, "Supersonic exit",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"));
}

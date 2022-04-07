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
             py::arg("terms"),
             py::arg("tme"));
  inlets.def("cubicSplineSubsonicInlet", &cubicSplineSubsonicInlet, "Cubic spline subsonic inlet",
             py::arg("block_ object"),
             py::arg("face_ object"),
             py::arg("eos pointer"),
             py::arg("thtrdat_ object"),
             py::arg("terms"),
             py::arg("tme"));
  inlets.def("supersonicInlet", &supersonicInlet, "Supersonic inlet",
             py::arg("block_ object"),
             py::arg("face_ object"),
             py::arg("eos pointer"),
             py::arg("thtrdat_ object"),
             py::arg("terms"),
             py::arg("tme"));
  inlets.def("constantMassFluxSubsonicInlet", &constantMassFluxSubsonicInlet, "Const mass flux subsonic inlet",
             py::arg("block_ object"),
             py::arg("face_ object"),
             py::arg("eos pointer"),
             py::arg("thtrdat_ object"),
             py::arg("terms"),
             py::arg("tme"));

  //  |----> walls.cpp
  py::module walls = bcs.def_submodule("walls", "walls boundary conditions module");
  walls.def("adiabaticNoSlipWall", &adiabaticNoSlipWall, "Adiabatic no slip wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"),
            py::arg("tme"));
  walls.def("adiabaticSlipWall", &adiabaticSlipWall, "Adiabatic slip wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"),
            py::arg("tme"));
  walls.def("adiabaticMovingWall", &adiabaticMovingWall, "Adiabatic moving wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"),
            py::arg("tme"));
  walls.def("isoTNoSlipWall", &isoTNoSlipWall, "IsoT no slip wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"),
            py::arg("tme"));
  walls.def("isoTSlipWall", &isoTSlipWall, "IsoT slip wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"),
            py::arg("tme"));
  walls.def("isoTMovingWall", &isoTMovingWall, "Iso thermal moving wall",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"),
            py::arg("tme"));

  //  |----> exits.cpp
  py::module exits = bcs.def_submodule("exits", "exit boundary conditions module");
  exits.def("constantPressureSubsonicExit", &constantPressureSubsonicExit, "Const pressure subsonic exit",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"),
            py::arg("tme"));
  exits.def("supersonicExit", &supersonicExit, "Supersonic exit",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"),
            py::arg("tme"));

  //  |----> periodics.cpp
  py::module periodics = bcs.def_submodule("periodics", "periodics boundary conditions module");
  periodics.def("periodicRotHigh", &periodicRotHigh, "High rotational periodic face",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"),
            py::arg("tme"));
  periodics.def("periodicRotLow", &periodicRotLow, "Low rotational periodic face",
            py::arg("block_ object"),
            py::arg("face_ object"),
            py::arg("eos pointer"),
            py::arg("thtrdat_ object"),
            py::arg("terms"),
            py::arg("tme"));
}

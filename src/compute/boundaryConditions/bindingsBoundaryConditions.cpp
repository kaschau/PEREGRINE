#include "boundaryConditions.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

void bindBoundaryConditions(nb::module_ &m) {
  // ./boundaryConditions
  nb::module_ bcs = m.def_submodule("bcs", "boundary conditions module");

  //  |----> inlets.cpp
  nb::module_ inlets =
      bcs.def_submodule("inlets", "inlet boundary conditions module");
  inlets.def("constantVelocitySubsonicInlet", &constantVelocitySubsonicInlet,
             "Const velo subsonic inlet", nb::arg("block_ object"),
             nb::arg("face_ object"), nb::arg("eos pointer"),
             nb::arg("thtrdat_ object"), nb::arg("terms"), nb::arg("tme"));
  inlets.def("cubicSplineSubsonicInlet", &cubicSplineSubsonicInlet,
             "Cubic spline subsonic inlet", nb::arg("block_ object"),
             nb::arg("face_ object"), nb::arg("eos pointer"),
             nb::arg("thtrdat_ object"), nb::arg("terms"), nb::arg("tme"));
  inlets.def("supersonicInlet", &supersonicInlet, "Supersonic inlet",
             nb::arg("block_ object"), nb::arg("face_ object"),
             nb::arg("eos pointer"), nb::arg("thtrdat_ object"),
             nb::arg("terms"), nb::arg("tme"));
  inlets.def("constantMassFluxSubsonicInlet", &constantMassFluxSubsonicInlet,
             "Const mass flux subsonic inlet", nb::arg("block_ object"),
             nb::arg("face_ object"), nb::arg("eos pointer"),
             nb::arg("thtrdat_ object"), nb::arg("terms"), nb::arg("tme"));
  inlets.def("stagnationSubsonicInlet", &stagnationSubsonicInlet,
             "Stagnation subsonic inlet", nb::arg("block_ object"),
             nb::arg("face_ object"), nb::arg("eos pointer"),
             nb::arg("thtrdat_ object"), nb::arg("terms"), nb::arg("tme"));

  //  |----> walls.cpp
  nb::module_ walls =
      bcs.def_submodule("walls", "walls boundary conditions module");
  walls.def("adiabaticNoSlipWall", &adiabaticNoSlipWall,
            "Adiabatic no slip wall", nb::arg("block_ object"),
            nb::arg("face_ object"), nb::arg("eos pointer"),
            nb::arg("thtrdat_ object"), nb::arg("terms"), nb::arg("tme"));
  walls.def("adiabaticSlipWall", &adiabaticSlipWall, "Adiabatic slip wall",
            nb::arg("block_ object"), nb::arg("face_ object"),
            nb::arg("eos pointer"), nb::arg("thtrdat_ object"),
            nb::arg("terms"), nb::arg("tme"));
  walls.def("adiabaticMovingWall", &adiabaticMovingWall,
            "Adiabatic moving wall", nb::arg("block_ object"),
            nb::arg("face_ object"), nb::arg("eos pointer"),
            nb::arg("thtrdat_ object"), nb::arg("terms"), nb::arg("tme"));
  walls.def("isoTNoSlipWall", &isoTNoSlipWall, "IsoT no slip wall",
            nb::arg("block_ object"), nb::arg("face_ object"),
            nb::arg("eos pointer"), nb::arg("thtrdat_ object"),
            nb::arg("terms"), nb::arg("tme"));
  walls.def("isoTSlipWall", &isoTSlipWall, "IsoT slip wall",
            nb::arg("block_ object"), nb::arg("face_ object"),
            nb::arg("eos pointer"), nb::arg("thtrdat_ object"),
            nb::arg("terms"), nb::arg("tme"));
  walls.def("isoTMovingWall", &isoTMovingWall, "Iso thermal moving wall",
            nb::arg("block_ object"), nb::arg("face_ object"),
            nb::arg("eos pointer"), nb::arg("thtrdat_ object"),
            nb::arg("terms"), nb::arg("tme"));

  //  |----> exits.cpp
  nb::module_ exits =
      bcs.def_submodule("exits", "exit boundary conditions module");
  exits.def("constantPressureSubsonicExit", &constantPressureSubsonicExit,
            "Const pressure subsonic exit", nb::arg("block_ object"),
            nb::arg("face_ object"), nb::arg("eos pointer"),
            nb::arg("thtrdat_ object"), nb::arg("terms"), nb::arg("tme"));
  exits.def("supersonicExit", &supersonicExit, "Supersonic exit",
            nb::arg("block_ object"), nb::arg("face_ object"),
            nb::arg("eos pointer"), nb::arg("thtrdat_ object"),
            nb::arg("terms"), nb::arg("tme"));

  //  |----> periodics.cpp
  nb::module_ periodics =
      bcs.def_submodule("periodics", "periodics boundary conditions module");
  periodics.def("periodicRotHigh", &periodicRotHigh,
                "High rotational periodic face", nb::arg("block_ object"),
                nb::arg("face_ object"), nb::arg("eos pointer"),
                nb::arg("thtrdat_ object"), nb::arg("terms"), nb::arg("tme"));
  periodics.def("periodicRotLow", &periodicRotLow,
                "Low rotational periodic face", nb::arg("block_ object"),
                nb::arg("face_ object"), nb::arg("eos pointer"),
                nb::arg("thtrdat_ object"), nb::arg("terms"), nb::arg("tme"));
}

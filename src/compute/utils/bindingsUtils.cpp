#include "utils.hpp"
#include <kokkosTypes.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void bindUtils(nb::module_ &m) {
  // ./utils
  nb::module_ utils = m.def_submodule("utils", "utility module");
  //  |----> applyFluxes.cpp
  utils.def("applyFlux", &applyFlux, "Apply flux directly",
            nb::arg("block_ object"), nb::arg(""));
  utils.def("applyHybridFlux", &applyHybridFlux, "Blend flux with another",
            nb::arg("block_ object"), nb::arg("primary"));
  utils.def("applyDissipationFlux", &applyDissipationFlux,
            "Apply artificial dissipation flux", nb::arg("block_ object"),
            nb::arg(""));
  //  |----> dQzero.cpp
  utils.def("dQzero", &dQzero, "Zero out dQ array", nb::arg("block_ object"));
  //  |----> dq2FD.cpp
  utils.def("dq2FD", &dq2FD,
            "Second order approx of spatial derivative of q array via finite "
            "difference",
            nb::arg("block_ object"));
  //  |----> dq4FD.cpp
  utils.def("dq4FD", &dq4FD,
            "Fourth order approx of spatial derivative of q array via finite "
            "difference",
            nb::arg("block_ object"));
  //    |------> axpby
  utils.def("AEQConst",
            nb::overload_cast<fourDview &, const double &>(&AEQConst),
            "A = Const", nb::arg("A fourDview"), nb::arg("const double Const"));
  utils.def(
      "AEQConst", nb::overload_cast<threeDview &, const double &>(&AEQConst),
      "A = Const", nb::arg("A threeDview"), nb::arg("const double Const"));
  utils.def("AEQB", &AEQB, "A = B", nb::arg("A view"), nb::arg("B view"));
  utils.def("ApEQxB", &ApEQxB, "A += x*B", nb::arg("A view"),
            nb::arg("x double"), nb::arg("B view"));
  utils.def("AEQxB", &AEQxB, "A = x*B", nb::arg("A view"), nb::arg("x double"),
            nb::arg("B view"));
  utils.def("CEQxApyB", &CEQxApyB, "C = x*A + y*B", nb::arg("C view"),
            nb::arg("x double"), nb::arg("A view"), nb::arg("y double"),
            nb::arg("B view"));
  //    |------> cfl
  utils.def("CFLmax", &CFLmax,
            "Find max acoustic, convective, spectral radius CFL factors c/dx",
            nb::arg("block_ object"));
  //    |------> checkNan
  utils.def("checkNan", &checkNan, "Check for any nans/infs in the Q array",
            nb::arg("std::vector<block_ object>"));
  //    |------> sendRecvBuffer
  utils.def("extractSendBuffer",
            nb::overload_cast<threeDview &, threeDview &, face_ &,
                              const std::vector<int> &>(&extractSendBuffer),
            "Extract the send buffer of a view", nb::arg("kokkos view"),
            nb::arg("buffer"), nb::arg("face object"),
            nb::arg("lists of slices"));
  utils.def("extractSendBuffer",
            nb::overload_cast<fourDview &, fourDview &, face_ &,
                              const std::vector<int> &>(&extractSendBuffer),
            "Extract the send buffer of a view", nb::arg("kokkos view"),
            nb::arg("buffer"), nb::arg("face object"),
            nb::arg("lists of slices"));
  utils.def("placeRecvBuffer",
            nb::overload_cast<threeDview &, threeDview &, face_ &,
                              const std::vector<int> &>(&placeRecvBuffer),
            "Place the recv buffer of a view", nb::arg("kokkos view"),
            nb::arg("buffer"), nb::arg("face object"),
            nb::arg("lists of slices"));
  utils.def("placeRecvBuffer",
            nb::overload_cast<fourDview &, fourDview &, face_ &,
                              const std::vector<int> &>(&placeRecvBuffer),
            "Place the recv buffer of a view", nb::arg("kokkos view"),
            nb::arg("buffer"), nb::arg("face object"),
            nb::arg("lists of slices"));
  //    |------> viscousSponge
  utils.def("viscousSponge", &viscousSponge, "Compute viscous multiplier",
            nb::arg("block_"), nb::arg("origin"), nb::arg("ending"),
            nb::arg("mult"));
}

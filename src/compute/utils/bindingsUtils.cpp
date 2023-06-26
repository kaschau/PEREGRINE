#include "block_.hpp"
#include "thtrdat_.hpp"
#include "utils.hpp"
#include <kokkosTypes.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindUtils(py::module_ &m) {
  // ./utils
  py::module utils = m.def_submodule("utils", "utility module");
  //  |----> applyFluxes.cpp
  utils.def("applyFlux", &applyFlux, "Apply flux directly",
            py::arg("block_ object"), py::arg(""), py::arg(""));
  utils.def("applyHybridFlux", &applyHybridFlux, "Blend flux with another",
            py::arg("block_ object"), py::arg(""), py::arg("primary"));
  utils.def("applyDissipationFlux", &applyDissipationFlux,
            "Apply artificial dissipation flux", py::arg("block_ object"),
            py::arg(""), py::arg(""));
  //  |----> dQzero.cpp
  utils.def("dQzero", &dQzero, "Zero out dQ array", py::arg("block_ object"));
  //  |----> dq2FD.cpp
  utils.def("dq2FD", &dq2FD,
            "Second order approx of spatial derivative of q array via finite "
            "difference",
            py::arg("block_ object"));
  //  |----> dq4FD.cpp
  utils.def("dq4FD", &dq4FD,
            "Fourth order approx of spatial derivative of q array via finite "
            "difference",
            py::arg("block_ object"));
  //    |------> axpby
  utils.def("AEQConst",
            py::overload_cast<fourDview &, const double &>(&AEQConst),
            "A = Const", py::arg("A fourDview"), py::arg("const double Const"));
  utils.def(
      "AEQConst", py::overload_cast<threeDview &, const double &>(&AEQConst),
      "A = Const", py::arg("A threeDview"), py::arg("const double Const"));
  utils.def("AEQB", &AEQB, "A = B", py::arg("A view"), py::arg("B view"));
  utils.def("ApEQxB", &ApEQxB, "A += x*B", py::arg("A view"),
            py::arg("x double"), py::arg("B view"));
  utils.def("AEQxB", &AEQxB, "A = x*B", py::arg("A view"), py::arg("x double"),
            py::arg("B view"));
  utils.def("CEQxApyB",
            py::overload_cast<fourDview &, const double &, const fourDview &,
                              const double &, const fourDview &>(&CEQxApyB),
            "C = x*A + y*B", py::arg("C view"), py::arg("x double"),
            py::arg("A view"), py::arg("y double"), py::arg("B view"));
  utils.def("CEQxApyB",
            py::overload_cast<threeDview &, const double &, const threeDview &,
                              const double &, const threeDview &>(&CEQxApyB),
            "C = x*A + y*B", py::arg("C view"), py::arg("x double"),
            py::arg("A view"), py::arg("y double"), py::arg("B view"));
  //    |------> cfl
  utils.def("CFLmax", &CFLmax,
            "Find max acoustic, convective, spectral radius CFL factors c/dx",
            py::arg("block_ object"));
  //    |------> checkNan
  utils.def("checkNan", &checkNan, "Check for any nans/infs in the Q array",
            py::arg("std::vector<block_ object>"));
  //    |------> sendRecvBuffer
  utils.def("extractSendBuffer",
            py::overload_cast<threeDview &, threeDview &, face_ &,
                              const std::vector<int> &>(&extractSendBuffer),
            "Extract the send buffer of a view", py::arg("kokkos view"),
            py::arg("buffer"), py::arg("face object"),
            py::arg("lists of slices"));
  utils.def("extractSendBuffer",
            py::overload_cast<fourDview &, fourDview &, face_ &,
                              const std::vector<int> &>(&extractSendBuffer),
            "Extract the send buffer of a view", py::arg("kokkos view"),
            py::arg("buffer"), py::arg("face object"),
            py::arg("lists of slices"));
  utils.def("placeRecvBuffer",
            py::overload_cast<threeDview &, threeDview &, face_ &,
                              const std::vector<int> &>(&placeRecvBuffer),
            "Place the recv buffer of a view", py::arg("kokkos view"),
            py::arg("buffer"), py::arg("face object"),
            py::arg("lists of slices"));
  utils.def("placeRecvBuffer",
            py::overload_cast<fourDview &, fourDview &, face_ &,
                              const std::vector<int> &>(&placeRecvBuffer),
            "Place the recv buffer of a view", py::arg("kokkos view"),
            py::arg("buffer"), py::arg("face object"),
            py::arg("lists of slices"));
  //    |------> viscousSponge
  utils.def("viscousSponge", &viscousSponge, "Compute viscous multiplier",
            py::arg("block_"), py::arg("origin"), py::arg("ending"),
            py::arg("mult"));
  //    |------> computeEntropy
  utils.def("computeEntropy", &computeEntropy,
            "Find max acoustic, convective, spectral radius CFL factors c/dx",
            py::arg("mb object"), py::arg("mb object"));
  //    |------> sumEntropy
  utils.def("sumEntropy", &sumEntropy,
            "Find max acoustic, convective, spectral radius CFL factors c/dx",
            py::arg("mb object"));
  //    |------> etropy
  utils.def("entropy", &entropy,
            "Find max acoustic, convective, spectral radius CFL factors c/dx",
            py::arg("block object"), py::arg("thtrdat"));
}

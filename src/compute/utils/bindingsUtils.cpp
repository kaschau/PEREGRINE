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
            py::arg("block_ object"), py::arg(""));
  utils.def("applyHybridFlux", &applyHybridFlux, "Blend flux with another",
            py::arg("block_ object"), py::arg("primary"));
  utils.def("applyDissipationFlux", &applyDissipationFlux,
            "Apply artificial dissipation flux", py::arg("block_ object"),
            py::arg(""));
  //  |----> dQzero.cpp
  utils.def("dQzero", &dQzero, "Zero out dQ array", py::arg("block_ object"));
  //  |----> dq2FD.cpp
  utils.def("dq2FD", &dq2FD,
            "Second order approx of spatial derivative of q array via finite "
            "difference",
            py::arg("block_ object"));
  //    |------> axpby
  utils.def("AEQB", &AEQB, "A = B", py::arg("A view"), py::arg("B view"));
  utils.def("axnpby",
            py::overload_cast<fourDview &, const double &, const double &,
                              const fourDview &>(&axnpby),
            "A = a*A + b*B", py::arg("A"), py::arg("a"), py::arg("b"),
            py::arg("B"));
  utils.def(
      "axnpby",
      py::overload_cast<fourDview &, const double &, const double &,
                        const fourDview &, const double &, const fourDview &>(
          &axnpby),
      "A = a*A + b*B + c*C", py::arg("A"), py::arg("a"), py::arg("b"),
      py::arg("B"), py::arg("c"), py::arg("C"));
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
}

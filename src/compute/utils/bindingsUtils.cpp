#include "compute.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindUtils(py::module_ &m) {
  // ./utils
  py::module utils = m.def_submodule("utils", "utility module");
  //  |----> applyFluxes.cpp
  utils.def("applyFlux", &applyFlux, "Apply flux directly",
        py::arg("block_ object"),
        py::arg(""));
  utils.def("applyHybridFlux", &applyHybridFlux, "Blend flux with another",
        py::arg("block_ object"),
        py::arg("primary"));
  utils.def("applyDissipationFlux", &applyDissipationFlux, "Apply artificial dissipation flux",
        py::arg("block_ object"),
        py::arg(""));
  //  |----> dQzero.cpp
  utils.def("dQzero", &dQzero, "Zero out dQ array",
        py::arg("block_ object"));
  //  |----> dq2FD.cpp
  utils.def("dq2FD", &dq2FD, "Second order approx of spatial derivative of q array via finite difference",
        py::arg("block_ object"));
  //  |----> dq4FD.cpp
  utils.def("dq4FD", &dq4FD, "Fourth order approx of spatial derivative of q array via finite difference",
        py::arg("block_ object"));
  //    |------> axpby
  utils.def("AEQB", &AEQB, "A = B",
        py::arg("A view"),
        py::arg("B view"));
  utils.def("ApEQxB", &ApEQxB, "A += x*B",
        py::arg("A view"),
        py::arg("x double"),
        py::arg("B view"));
  utils.def("AEQxB", &AEQxB, "A = x*B",
        py::arg("A view"),
        py::arg("x double"),
        py::arg("B view"));
  utils.def("CEQxApyB", &CEQxApyB, "C = x*A + y*B",
        py::arg("C view"),
        py::arg("x double"),
        py::arg("A view"),
        py::arg("y double"),
        py::arg("B view"));
  //    |------> cfl
  utils.def("CFLmax", &CFLmax, "Find max acoustic, convective, spectral radius CFL factors c/dx",
        py::arg("block_ object"));
  //    |------> checkNan
  utils.def("checkNan", &checkNan, "Check for any nans/infs in the Q array",
        py::arg("std::vector<block_ object>"));
  //    |------> sendRecvBuffer
  utils.def("extract_sendBuffer3", &extract_sendBuffer3, "Extract the send buffer of a view",
        py::arg("kokkos view"),
        py::arg("face object"),
        py::arg("lists of slices"));
  utils.def("extract_sendBuffer4", &extract_sendBuffer4, "Extract the send buffer of a view",
        py::arg("kokkos view"),
        py::arg("face object"),
        py::arg("lists of slices"));
  utils.def("place_recvBuffer3", &place_recvBuffer3, "Place the recv buffer of a view",
        py::arg("kokkos view"),
        py::arg("face object"),
        py::arg("lists of slices"));
  utils.def("place_recvBuffer4", &place_recvBuffer4, "Place the recv buffer of a view",
        py::arg("kokkos view"),
        py::arg("face object"),
        py::arg("lists of slices"));
}

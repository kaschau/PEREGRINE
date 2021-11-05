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
        py::arg("primary"));
  utils.def("applyHybridFlux", &applyHybridFlux, "Blend flux with another",
        py::arg("block_ object"),
        py::arg("primary"));
  utils.def("applyDissipationFlux", &applyDissipationFlux, "Apply artificial dissipation flux",
        py::arg("block_ object"),
        py::arg("primary"));
  //  |----> dQzero.cpp
  utils.def("dQzero", &dQzero, "Zero out dQ array",
        py::arg("block_ object"));
  //  |----> dq2FD.cpp
  utils.def("dq2FD", &dq2FD, "Second order approx of spatial derivative of q array via finite difference",
        py::arg("block_ object"));
  //  |----> dq4FD.cpp
  utils.def("dq4FD", &dq4FD, "Fourth order approx of spatial derivative of q array via finite difference",
        py::arg("block_ object"));

}
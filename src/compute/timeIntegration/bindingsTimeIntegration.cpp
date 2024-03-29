#include "timeIntegration.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindTimeIntegration(py::module_ &m) {
  // ./timeIntegration
  py::module timeIntegration =
      m.def_submodule("timeIntegration", "Time integration module");
  //  |----> dualTime.cpp
  timeIntegration.def("dQdt", &dQdt, "Real time derivative source term",
                      py::arg("block_"), py::arg("dt"));
  timeIntegration.def("localDtau", &localDtau, "Local pseudo time step",
                      py::arg("block_"), py::arg("viscous"));
  timeIntegration.def("DTrk3s1", &DTrk3s1, "Dual Time rk3 stage 1",
                      py::arg("block_"));
  timeIntegration.def("DTrk3s2", &DTrk3s2, "Dual Time rk3 stage 2",
                      py::arg("block_"));
  timeIntegration.def("DTrk3s3", &DTrk3s3, "Dual Time rk3 stage 3",
                      py::arg("block_"));
  timeIntegration.def("residual", &residual, "Residual", py::arg("mb"));
  timeIntegration.def("invertDQ", &invertDQ, "Solve dq = \\Gamma^{-1} dQ",
                      py::arg("block_"), py::arg("dt"), py::arg("thtrdat_"),
                      py::arg("viscous"));
  //  |----> maccormack.cpp
  timeIntegration.def("corrector", &corrector, "maccormack corrector",
                      py::arg("block_"), py::arg("dt"));
  //  |----> rk2Stages.cpp
  timeIntegration.def("rk2s1", &rk2s1, "rk2 stage 1", py::arg("block_"),
                      py::arg("dt"));
  timeIntegration.def("rk2s2", &rk2s2, "rk2 stage 2", py::arg("block_"),
                      py::arg("dt"));
  //  |----> rk3Stages.cpp
  timeIntegration.def("rk3s1", &rk3s1, "rk3 stage 1", py::arg("block_"),
                      py::arg("dt"));
  timeIntegration.def("rk3s2", &rk3s2, "rk3 stage 2", py::arg("block_"),
                      py::arg("dt"));
  timeIntegration.def("rk3s3", &rk3s3, "rk3 stage 3", py::arg("block_"),
                      py::arg("dt"));
  //  |----> rk34Stages.cpp
  timeIntegration.def("rk34s1", &rk34s1, "rk34 stage 1", py::arg("block_"),
                      py::arg("dt"));
  timeIntegration.def("rk34s2", &rk34s2, "rk34 stage 2", py::arg("block_"),
                      py::arg("dt"));
  timeIntegration.def("rk34s3", &rk34s3, "rk34 stage 3", py::arg("block_"),
                      py::arg("dt"));
  timeIntegration.def("rk34s4", &rk34s4, "rk34 stage 4", py::arg("block_"),
                      py::arg("dt"));
  //  |----> rk4Stages.cpp
  timeIntegration.def("rk4s1", &rk4s1, "rk4 stage 1", py::arg("block_"),
                      py::arg("dt"));
  timeIntegration.def("rk4s2", &rk4s2, "rk4 stage 2", py::arg("block_"),
                      py::arg("dt"));
  timeIntegration.def("rk4s3", &rk4s3, "rk4 stage 3", py::arg("block_"),
                      py::arg("dt"));
  timeIntegration.def("rk4s4", &rk4s4, "rk4 stage 4", py::arg("block_"),
                      py::arg("dt"));
}

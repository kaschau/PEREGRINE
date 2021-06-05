#include "user.hpp"
#include <pybind11/pybind11.h>
#include <cstdlib>

namespace py = pybind11;

//--------------------------------------------------------------------------------------//
//
//        The python module
//
//--------------------------------------------------------------------------------------//

PYBIND11_MODULE(perepute, m) {
  ///
  /// This is a python binding to the user-defined generate_view function
  /// declared in user.hpp which returns a Kokkos::View. This function is called
  /// from ex-numpy.py
  ///
  //m.def("generate_view", &generate_view, "Generate a random view",
  //       py::arg("n") = 10);

  m.def("add", &add, "Add a float to entire view",
         py::arg("view"), py::arg("adder"));
  m.def("add2", &add2, "Add a float to entire view",
         py::arg("view"), py::arg("adder"),
         py::arg("imin"), py::arg("jmin"), py::arg("kmin"),
         py::arg("imax"), py::arg("jmax"), py::arg("kmax"));
  m.def("add3", &add3, "Add a float to entire view",
         py::arg("class"), py::arg("adder"));

  py::class_<block>(m, "block", py::dynamic_attr())
    .def(py::init<>())

    .def_readwrite("nblki", &block::nblki)

    .def_readwrite("nx", &block::nx)
    .def_readwrite("ny", &block::ny)
    .def_readwrite("nz", &block::nz)

    .def_readwrite("x", &block::x)
    .def_readwrite("y", &block::y)
    .def_readwrite("z", &block::z);

  static auto _atexit = []() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
  };

  atexit(_atexit);
}


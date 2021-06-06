#include "peregrine_computes.hpp"
#include "kokkos2peregrine.hpp"
#include "block.hpp"
#include <pybind11/pybind11.h>
#include <cstdlib>

namespace py = pybind11;

//--------------------------------------------------------------------------------------//
//
//        The python module
//
//--------------------------------------------------------------------------------------//

PYBIND11_MODULE(peregrine, m) {
  m.doc() = "Module to expose compute units written in C++ with Kokkos";
  m.attr("KokkosLocation") = &KokkosLocation;
  ///
  /// This is a python binding to the user-defined generate_view function
  /// declared in user.hpp which returns a Kokkos::View. This function is called
  /// from ex-numpy.py
  ///

  m.def("add3", &add3, "Add a float to entire threeDview",
         py::arg("Block object"), py::arg("Float to add"));

  py::class_<block>(m, "block", py::dynamic_attr())
    .def(py::init<>())

    .def_readwrite("nblki", &block::nblki)

    .def_readwrite("ni", &block::ni)
    .def_readwrite("nj", &block::nj)
    .def_readwrite("nk", &block::nk)

    .def_readwrite("x", &block::x)
    .def_readwrite("y", &block::y)
    .def_readwrite("z", &block::z);

  static auto _atexit = []() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
  };

  atexit(_atexit);
}


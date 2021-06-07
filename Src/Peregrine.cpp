#include "compute.hpp"
#include "kokkos2peregrine.hpp"
#include "Block.hpp"
#include <pybind11/pybind11.h>
#include <cstdlib>

namespace py = pybind11;

//--------------------------------------------------------------------------------------//
//
//        The python module
//
//--------------------------------------------------------------------------------------//

PYBIND11_MODULE(Peregrine, m) {
  m.doc() = "Module to expose compute units written in C++ with Kokkos";
  m.attr("KokkosLocation") = &KokkosLocation;
  ///
  /// This is a python binding to the user-defined generate_view function
  /// declared in user.hpp which returns a Kokkos::View. This function is called
  /// from ex-numpy.py
  ///

  m.def("add3", &add3, "Add a float to entire threeDview",
         py::arg("Block object"), py::arg("Float to add"));

  m.def("finalize_kokkos", &finalize_kokkos, "finalize kokkos");

  m.def("gen3Dview", &gen3Dview, "Generate a threeDview",
         py::arg("name"), py::arg("ni"), py::arg("nk"),py::arg("nk"));
  m.def("gen4Dview", &gen4Dview, "Generate a fourDview",
         py::arg("name"), py::arg("ni"), py::arg("nk"),py::arg("nk"), py::arg("nl"));


  py::class_<Block>(m, "Block", py::dynamic_attr())
    .def(py::init<>())

    .def_readwrite("nblki", &Block::nblki)

    .def_readwrite("ni", &Block::ni)
    .def_readwrite("nj", &Block::nj)
    .def_readwrite("nk", &Block::nk)

    .def_readwrite("ns", &Block::ns)

    // Grid Arrays
    .def_readwrite("x", &Block::x)
    .def_readwrite("y", &Block::y)
    .def_readwrite("z", &Block::z)

    // Conserved Array
    .def_readwrite("Qv", &Block::Qv)

    // Primtive Array
    .def_readwrite("T", &Block::T)
    .def_readwrite("P", &Block::p);

  static auto _atexit = []() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
  };

  atexit(_atexit);
}


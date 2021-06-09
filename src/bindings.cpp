
//#include "kokkos_types.hpp"
//#include "Block.hpp"
#include "Kokkos_Core.hpp"
#include "compute.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

//--------------------------------------------------------------------------------------//
//
//        The python module
//
//--------------------------------------------------------------------------------------//

PYBIND11_MODULE(compute_, m) {
  m.doc() = "Module to expose compute units written in C++ with Kokkos";
  m.attr("KokkosLocation") = &KokkosLocation;

////////////////////////////////////////////////////////////////////////////////
///////////////////  C++ Parent block_ class ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
  py::class_<block_>(m, "block_", py::dynamic_attr())
    .def(py::init<>())

    .def_readwrite("nblki", &block_::nblki)

    .def_readwrite("ni", &block_::ni)
    .def_readwrite("nj", &block_::nj)
    .def_readwrite("nk", &block_::nk)

    .def_readwrite("ns", &block_::ns)

    // Grid Arrays
    .def_readwrite("x_", &block_::x_)
    .def_readwrite("y_", &block_::y_)
    .def_readwrite("z_", &block_::z_)

    // Conserved Array
    .def_readwrite("Q_", &block_::Q_)
    // Primative Array
    .def_readwrite("q_", &block_::q_);


  // Temporary creation stuff
  m.def("gen3Dview", &gen3Dview, "Generate a threeDview",
        py::arg("name"), py::arg("ni"), py::arg("nk"),py::arg("nk"));
  m.def("gen4Dview", &gen4Dview, "Generate a fourDview",
        py::arg("name"), py::arg("ni"), py::arg("nk"),py::arg("nk"), py::arg("nl"));

  m.def("finalize_kokkos", &finalize_kokkos, "finalize kokkos");

////////////////////////////////////////////////////////////////////////////////
///////////////////////////  Compute Functions /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  // ./Grid
  //  |----> metrics
  m.def("metrics", &metrics, "Compute grid metrics on primary grid",
        py::arg("block_ object"));



  m.def("add3D", &add3D, "Add a float to entire threeDview",
        py::arg("block_ object"), py::arg("Float to add"));

  static auto _atexit = []() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
  };

  atexit(_atexit);
}

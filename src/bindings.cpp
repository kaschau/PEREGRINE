
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
    .def_readwrite("x", &block_::x)
    .def_readwrite("y", &block_::y)
    .def_readwrite("z", &block_::z)

    // Grid Metrics
    // Cell Centers
    .def_readwrite("xc", &block_::xc)
    .def_readwrite("yc", &block_::yc)
    .def_readwrite("zc", &block_::zc)

    // i face area vector
    .def_readwrite("isx", &block_::isx)
    .def_readwrite("isy", &block_::isy)
    .def_readwrite("isz", &block_::isz)
    .def_readwrite("iS" , &block_::iS )

    // j face area vector
    .def_readwrite("jsx", &block_::jsx)
    .def_readwrite("jsy", &block_::jsy)
    .def_readwrite("jsz", &block_::jsz)
    .def_readwrite("jS" , &block_::jS )

    // k face area vector
    .def_readwrite("ksx", &block_::ksx)
    .def_readwrite("ksy", &block_::ksy)
    .def_readwrite("ksz", &block_::ksz)
    .def_readwrite("kS" , &block_::kS );

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


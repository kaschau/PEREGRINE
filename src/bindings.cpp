
//#include "kokkos_types.hpp"
//#include "Block.hpp"
#include "Kokkos_Core.hpp"
#include "compute.hpp"
#include "block_.hpp"
#include "thermdat_.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

//--------------------------------------------------------------------------------------//
//
//        The python module
//
//--------------------------------------------------------------------------------------//

PYBIND11_MODULE(compute, m) {
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

    .def_readwrite("ne", &block_::ne)

//----------------------------------------------------------------------------//
//  Primary grid node coordinates
//----------------------------------------------------------------------------//
    .def_readwrite("x", &block_::x)
    .def_readwrite("y", &block_::y)
    .def_readwrite("z", &block_::z)

//----------------------------------------------------------------------------//
//  Primary metrics
//----------------------------------------------------------------------------//
    // Cell Centers
    .def_readwrite("xc", &block_::xc)
    .def_readwrite("yc", &block_::yc)
    .def_readwrite("zc", &block_::zc)
    .def_readwrite("J" , &block_::J )
    // i face area vector
    .def_readwrite("isx", &block_::isx)
    .def_readwrite("isy", &block_::isy)
    .def_readwrite("isz", &block_::isz)
    .def_readwrite("iS" , &block_::iS )
    .def_readwrite("inx", &block_::inx)
    .def_readwrite("iny", &block_::iny)
    .def_readwrite("inz", &block_::inz)
    // j face area vector
    .def_readwrite("jsx", &block_::jsx)
    .def_readwrite("jsy", &block_::jsy)
    .def_readwrite("jsz", &block_::jsz)
    .def_readwrite("jS" , &block_::jS )
    .def_readwrite("jnx", &block_::jnx)
    .def_readwrite("jny", &block_::jny)
    .def_readwrite("jnz", &block_::jnz)
    // k face area vector
    .def_readwrite("ksx", &block_::ksx)
    .def_readwrite("ksy", &block_::ksy)
    .def_readwrite("ksz", &block_::ksz)
    .def_readwrite("kS" , &block_::kS )
    .def_readwrite("knx", &block_::knx)
    .def_readwrite("kny", &block_::kny)
    .def_readwrite("knz", &block_::knz)

//----------------------------------------------------------------------------//
//  Flow variables
//----------------------------------------------------------------------------//
    // Conservative variables
    .def_readwrite("Q" , &block_::Q )
    .def_readwrite("q" , &block_::q )
    .def_readwrite("dQ", &block_::dQ )
    // Thermo variables
    .def_readwrite("qh", &block_::qh )

    // RK stages
    .def_readwrite("rhs0", &block_::rhs0 )
    .def_readwrite("rhs1", &block_::rhs1 )
    .def_readwrite("rhs2", &block_::rhs2 )
    .def_readwrite("rhs3", &block_::rhs3 )

    // Flux Arrays
    .def_readwrite("iF", &block_::iF )
    .def_readwrite("jF", &block_::jF )
    .def_readwrite("kF", &block_::kF );

////////////////////////////////////////////////////////////////////////////////
///////////////////  C++ Parent thermdat_ class ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  py::class_<thermdat_>(m, "thermdat_", py::dynamic_attr())
    .def(py::init<>())

    .def_readwrite("ns", &thermdat_::ns)
    .def_readwrite("Ru", &thermdat_::Ru)
    .def_readwrite("species_names", &thermdat_::species_names)
    .def_readwrite("MW", &thermdat_::MW)
    .def_readwrite("cp0", &thermdat_::cp0)
    .def_readwrite("N7", &thermdat_::N7);

////////////////////////////////////////////////////////////////////////////////
///////////////////////////  Compute Functions /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  // ./Grid
  //  |----> metrics
  m.def("metrics", &metrics, "Compute grid metrics on primary grid",
        py::arg("block_ object"));

  // ./flux
  //  |----> advective
  m.def("advective", &advective, "Compute centered difference flux",
        py::arg("block_ object"),
        py::arg("thermdat_ object"));
  //  |----> viscous
  m.def("diffusive", &diffusive, "Compute centered diffusive flux",
        py::arg("block_ object"),
        py::arg("thermdat_ object"));

  // ./thermo
  //  |----> cpg
  m.def("cpg", &cpg, "Update primatives or conservatives with cpg assumption",
        py::arg("block_ object"),
        py::arg("thermdat_ object"),
        py::arg("face"),
        py::arg("given"));
  //  |----> tpg
  m.def("tpg", &tpg, "Update primatives or conservatives with tpg assumption",
        py::arg("block_ object"),
        py::arg("thermdat_ object"),
        py::arg("face"),
        py::arg("given"));





  static auto _atexit = []() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
  };

  atexit(_atexit);
}


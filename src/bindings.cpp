#include "Kokkos_Core.hpp"
#include "compute.hpp"
#include "block_.hpp"
#include "face_.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

//--------------------------------------------------------------------------------------//
//
//        The python module
//
//--------------------------------------------------------------------------------------//

void bindAdvFlux(py::module_ &);
void bindBoundaryConditions(py::module_ &);
void bindChemistry(py::module_ &);
void bindDiffFlux(py::module_ &);
void bindSwitches(py::module_ &);
void bindThermo(py::module_ &);
void bindTransport(py::module_ &);
void bindTimeIntegration(py::module_ &);
void bindUtils(py::module_ &);

PYBIND11_MODULE(compute, m) {
  m.doc() = "Module to expose compute units written in C++ with Kokkos";
  m.attr("KokkosLocation") = &KokkosLocation;

  bindAdvFlux(m);
  bindBoundaryConditions(m);
  bindChemistry(m);
  bindDiffFlux(m);
  bindSwitches(m);
  bindThermo(m);
  bindTransport(m);
  bindTimeIntegration(m);
  bindUtils(m);

////////////////////////////////////////////////////////////////////////////////
///////////////////  C++ Parent block_ class ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  py::class_<block_>(m, "block_", py::dynamic_attr())
    .def(py::init<>())

    .def_readwrite("nblki", &block_::nblki)

    .def_readwrite("ni", &block_::ni)
    .def_readwrite("nj", &block_::nj)
    .def_readwrite("nk", &block_::nk)
    .def_readwrite("ng", &block_::ng)

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
    // Cell center metrics
    .def_readwrite("dEdx" , &block_::dEdx )
    .def_readwrite("dEdy" , &block_::dEdy )
    .def_readwrite("dEdz" , &block_::dEdz )
    .def_readwrite("dNdx" , &block_::dNdx )
    .def_readwrite("dNdy" , &block_::dNdy )
    .def_readwrite("dNdz" , &block_::dNdz )
    .def_readwrite("dXdx" , &block_::dXdx )
    .def_readwrite("dXdy" , &block_::dXdy )
    .def_readwrite("dXdz" , &block_::dXdz )
    // i Face centers
    .def_readwrite("ixc", &block_::ixc)
    .def_readwrite("iyc", &block_::iyc)
    .def_readwrite("izc", &block_::izc)
    // i face area vector
    .def_readwrite("isx", &block_::isx)
    .def_readwrite("isy", &block_::isy)
    .def_readwrite("isz", &block_::isz)
    .def_readwrite("iS" , &block_::iS )
    .def_readwrite("inx", &block_::inx)
    .def_readwrite("iny", &block_::iny)
    .def_readwrite("inz", &block_::inz)
    // j Face centers
    .def_readwrite("jxc", &block_::jxc)
    .def_readwrite("jyc", &block_::jyc)
    .def_readwrite("jzc", &block_::jzc)
    // j face area vector
    .def_readwrite("jsx", &block_::jsx)
    .def_readwrite("jsy", &block_::jsy)
    .def_readwrite("jsz", &block_::jsz)
    .def_readwrite("jS" , &block_::jS )
    .def_readwrite("jnx", &block_::jnx)
    .def_readwrite("jny", &block_::jny)
    .def_readwrite("jnz", &block_::jnz)
    // k Face centers
    .def_readwrite("kxc", &block_::kxc)
    .def_readwrite("kyc", &block_::kyc)
    .def_readwrite("kzc", &block_::kzc)
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
    // Conservative,primative variables
    .def_readwrite("Q" , &block_::Q )
    .def_readwrite("q" , &block_::q )
    .def_readwrite("dQ", &block_::dQ )

    // Spatial derivative of prim array
    .def_readwrite("dqdx", &block_::dqdx )
    .def_readwrite("dqdy", &block_::dqdy )
    .def_readwrite("dqdz", &block_::dqdz )

    // Thermo,transport variables
    .def_readwrite("qh", &block_::qh )
    .def_readwrite("qt", &block_::qt )

    // Chemistry
    .def_readwrite("omega", &block_::omega )

    // RK stages
    .def_readwrite("rhs0", &block_::rhs0 )
    .def_readwrite("rhs1", &block_::rhs1 )
    .def_readwrite("rhs2", &block_::rhs2 )
    .def_readwrite("rhs3", &block_::rhs3 )

    // Flux Arrays
    .def_readwrite("iF", &block_::iF )
    .def_readwrite("jF", &block_::jF )
    .def_readwrite("kF", &block_::kF )

    // Switch
    .def_readwrite("phi", &block_::phi );

  py::class_<face_>(m, "face_", py::dynamic_attr())
    .def(py::init<>())

    .def_readwrite("_nface", &face_::_nface)
    .def_readwrite("qBcVals", &face_::qBcVals)
    .def_readwrite("QBcVals", &face_::QBcVals);

  static auto _atexit = []() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
  };

  atexit(_atexit);
}



//#include "kokkos_types.hpp"
//#include "Block.hpp"
#include "Kokkos_Core.hpp"
#include "compute.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
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

////////////////////////////////////////////////////////////////////////////////
///////////////////////////  Compute Functions /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  // ./advFlux
  py::module advFlux = m.def_submodule("advFlux", "advective flux module");
  //  |----> secondOrderKEEP.cpp
  advFlux.def("secondOrderKEEP", &secondOrderKEEP, "Compute centeral difference euler fluxes via second order KEEP",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));
  //  |----> fourthOrderKEEP.cpp
  advFlux.def("fourthOrderKEEP", &fourthOrderKEEP, "Compute centeral difference euler fluxes via fourth order KEEP",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));
  //  |----> rusanov.cpp
  advFlux.def("rusanov", &rusanov, "Compute first order euler fluxes via rusanov",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));
  //  |----> ausmPlusUp.cpp
  advFlux.def("ausmPlusUp", &ausmPlusUp, "Compute inviscid fluxes via AUSM+UP",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));

  // ./diffFlux
  py::module diffFlux = m.def_submodule("diffFlux", "diffusive flux module");
  //  |----> diffusiveFlux.cpp
  diffFlux.def("diffusiveFlux", &diffusiveFlux, "Compute centeral difference viscous fluxes. Order set by dqdx",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"));

  // ./switches
  py::module switches = m.def_submodule("switches", "switches");
  //  |----> jameson.cpp
  switches.def("entropy", &entropy, "Compute switches based on entropy",
        py::arg("block_ object"));
  switches.def("pressure", &pressure, "Compute switches based on pressure",
        py::arg("block_ object"));
  //  |----> negateFluxes.cpp
  switches.def("noIFlux", &noIFlux, "Zero out primary flux via switch", py::arg("block_ object"));
  switches.def("noJFlux", &noJFlux, "Zero out primary flux via switch", py::arg("block_ object"));
  switches.def("noKFlux", &noKFlux, "Zero out primary flux via switch", py::arg("block_ object"));
  switches.def("noInoJFlux", &noInoJFlux, "Zero out primary flux via switch", py::arg("block_ object"));
  switches.def("noInoKFlux", &noInoKFlux, "Zero out primary flux via switch", py::arg("block_ object"));
  switches.def("noJnoKFlux", &noJnoKFlux, "Zero out primary flux via switch", py::arg("block_ object"));

  // ./thermo
  py::module thermo = m.def_submodule("thermo","thermo module");
  //  |----> cpg.cpp
  thermo.def("cpg", &cpg, "Update primatives or conservatives with cpg assumption",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"),
        py::arg("face"),
        py::arg("given"),
        py::arg("i")=0,
        py::arg("j")=0,
        py::arg("k")=0);
  //  |----> tpg.cpp
  thermo.def("tpg", &tpg, "Update primatives or conservatives with tpg assumption",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"),
        py::arg("face"),
        py::arg("given"),
        py::arg("i")=0,
        py::arg("j")=0,
        py::arg("k")=0);

  py::class_<thtrdat_>(thermo, "thtrdat_", py::dynamic_attr())
    .def(py::init<>())

    .def_readwrite("ns", &thtrdat_::ns)
    .def_readwrite("Ru", &thtrdat_::Ru)

    .def_readwrite("species_names", &thtrdat_::species_names)
    .def_readwrite("MW", &thtrdat_::MW)

    .def_readwrite("cp0", &thtrdat_::cp0)
    .def_readwrite("NASA7", &thtrdat_::NASA7)

    .def_readwrite("mu_poly", &thtrdat_::mu_poly)
    .def_readwrite("kappa_poly", &thtrdat_::kappa_poly)
    .def_readwrite("Dij_poly", &thtrdat_::Dij_poly);

  // ./transport
  py::module transport = m.def_submodule("transport", "transport module");
  //  |----> kineticTheory.cpp
  transport.def("kineticTheory", &kineticTheory, "Update transport properties from primatives via kinetic theory",
        py::arg("block_"),
        py::arg("thtrdat_ object"),
        py::arg("face"),
        py::arg("i")=0,
        py::arg("j")=0,
        py::arg("k")=0);

  // ./chemistry
  py::module chemistry = m.def_submodule("chemistry", "chemistry module");
  //  |----> CH4_O2_Stanford_Skeletal.cpp
  chemistry.def("chem_CH4_O2_Stanford_Skeletal", &chem_CH4_O2_Stanford_Skeletal, "Chemical source terms from",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"),
        py::arg("face")=0,
        py::arg("i")=0,
        py::arg("j")=0,
        py::arg("k")=0);
  //  |----> GRI30
  chemistry.def("chem_GRI30", &chem_GRI30, "Chemical source terms from GRI3.0",
        py::arg("block_ object"),
        py::arg("thtrdat_ object"),
        py::arg("face")=0,
        py::arg("i") = 0,
        py::arg("j") = 0,
        py::arg("k") = 0);

  // ./utils
  py::module utils = m.def_submodule("utils", "utility module");
  //  |----> applyFluxes.cpp
  utils.def("applyFlux", &applyFlux, "Apply flux directly",
        py::arg("block_ object"),
        py::arg("primary"));
  utils.def("hybridFlux", &hybridFlux, "Blend flux with another",
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


  static auto _atexit = []() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
  };

  atexit(_atexit);
}


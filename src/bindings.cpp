#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
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
void bindSubgrid(py::module_ &);
void bindSwitches(py::module_ &);
void bindThermo(py::module_ &);
void bindTransport(py::module_ &);
void bindTimeIntegration(py::module_ &);
void bindUtils(py::module_ &);
void bindKokkos(py::module_ &);

PYBIND11_MODULE(compute, m) {
  m.doc() = "Module to expose compute units written in C++ with Kokkos, as "
            "well as view/mirror/array exchange.";

  bindAdvFlux(m);
  bindBoundaryConditions(m);
  bindChemistry(m);
  bindDiffFlux(m);
  bindSubgrid(m);
  bindSwitches(m);
  bindThermo(m);
  bindTransport(m);
  bindTimeIntegration(m);
  bindUtils(m);
  bindKokkos(m);

  // --------------------------------------------------------------------------//
  // C++ Parent block_ class
  // --------------------------------------------------------------------------//

  py::class_<block_>(m, "block_", py::dynamic_attr())
      .def(py::init<>())

      .def_readwrite("nblki", &block_::nblki)

      .def_readwrite("ni", &block_::ni)
      .def_readwrite("nj", &block_::nj)
      .def_readwrite("nk", &block_::nk)
      .def_readwrite("ng", &block_::ng)

      .def_readwrite("ne", &block_::ne)
#ifdef NSCOMPILE
      .def_readonly("ns", &block_::ns)
#endif

      //----------------------------------------------------------------------------//
      //  Primary grid node coordinates
      //----------------------------------------------------------------------------//
      .def_readwrite("nodes", &block_::nodes)

      //----------------------------------------------------------------------------//
      //  Primary metrics
      //----------------------------------------------------------------------------//
      // Cell Centers
      .def_readwrite("cells", &block_::cells)
      .def_readwrite("J", &block_::J)
      // Cell lengths
      .def_readwrite("dIJK", &block_::dIJK)
      // Cell center metrics
      .def_readwrite("dENCdxyz", &block_::dENCdxyz)
      // i Face centers
      .def_readwrite("iFaces", &block_::iFaces)
      // i face area vector
      .def_readwrite("iS", &block_::iS)
      // j Face centers
      .def_readwrite("jFaces", &block_::jFaces)
      // j face area vector
      .def_readwrite("jS", &block_::jS)
      // k Face centers
      .def_readwrite("kFaces", &block_::kFaces)
      // k face area vector
      .def_readwrite("kS", &block_::kS)

      //----------------------------------------------------------------------------//
      //  Flow variables
      //----------------------------------------------------------------------------//
      // Conservative,primative variables
      .def_readwrite("Q", &block_::Q)
      .def_readwrite("q", &block_::q)
      .def_readwrite("dQ", &block_::dQ)

      // Spatial derivative of prim array
      .def_readwrite("grads", &block_::grads)

      // Thermo,transport variables
      .def_readwrite("qh", &block_::qh)
      .def_readwrite("qt", &block_::qt)

      // Chemistry
      .def_readwrite("omega", &block_::omega)

      // Time Integration Storage
      .def_readwrite("Q0", &block_::Q0)
      .def_readwrite("Q1", &block_::Q1)
      .def_readwrite("Q2", &block_::Q2)
      .def_readwrite("Q3", &block_::Q3)
      .def_readwrite("Qn", &block_::Qn)
      .def_readwrite("Qnm1", &block_::Qnm1)
      .def_readwrite("dtau", &block_::dtau)

      // Flux Arrays
      .def_readwrite("iF", &block_::iF)
      .def_readwrite("jF", &block_::jF)
      .def_readwrite("kF", &block_::kF)

      // Switch
      .def_readwrite("phi", &block_::phi);

  // --------------------------------------------------------------------------//
  // C++ Parent face_ class
  // --------------------------------------------------------------------------//
  py::class_<face_>(m, "face_", py::dynamic_attr())
      .def(py::init<>())

      .def_readwrite("nface", &face_::nface)
      .def_readwrite("orientTranspose", &face_::orientTranspose)
      .def_readwrite("orientFlip0", &face_::orientFlip0)
      .def_readwrite("orientFlip1", &face_::orientFlip1)

      .def_readwrite("qBcVals", &face_::qBcVals)
      .def_readwrite("QBcVals", &face_::QBcVals)
      .def_readwrite("sendBuffer_nodes", &face_::sendBuffer_nodes)
      .def_readwrite("sendBuffer_q", &face_::sendBuffer_q)
      .def_readwrite("sendBuffer_Q", &face_::sendBuffer_Q)
      .def_readwrite("sendBuffer_grads", &face_::sendBuffer_grads)
      .def_readwrite("sendBuffer_phi", &face_::sendBuffer_phi)

      .def_readwrite("recvBuffer_nodes", &face_::recvBuffer_nodes)
      .def_readwrite("recvBuffer_q", &face_::recvBuffer_q)
      .def_readwrite("recvBuffer_Q", &face_::recvBuffer_Q)
      .def_readwrite("recvBuffer_grads", &face_::recvBuffer_grads)
      .def_readwrite("recvBuffer_phi", &face_::recvBuffer_phi)

      .def_readwrite("periodicRotMatrix", &face_::periodicRotMatrix)

      ;

  static auto _atexit = []() {
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  };

  atexit(_atexit);
}

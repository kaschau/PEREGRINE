#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
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

PYBIND11_MODULE(compute, m) {
  m.doc() = "Module to expose compute units written in C++ with Kokkos";
  m.attr("KokkosLocation") = &KokkosLocation;

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

  ////////////////////////////////////////////////////////////////////////////////
  ///////////////////  C++ Parent block_ class
  //////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

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
      .def_readwrite("J", &block_::J)
      // Cell center metrics
      .def_readwrite("dEdx", &block_::dEdx)
      .def_readwrite("dEdy", &block_::dEdy)
      .def_readwrite("dEdz", &block_::dEdz)
      .def_readwrite("dNdx", &block_::dNdx)
      .def_readwrite("dNdy", &block_::dNdy)
      .def_readwrite("dNdz", &block_::dNdz)
      .def_readwrite("dCdx", &block_::dCdx)
      .def_readwrite("dCdy", &block_::dCdy)
      .def_readwrite("dCdz", &block_::dCdz)
      // i Face centers
      .def_readwrite("ixc", &block_::ixc)
      .def_readwrite("iyc", &block_::iyc)
      .def_readwrite("izc", &block_::izc)
      // i face area vector
      .def_readwrite("isx", &block_::isx)
      .def_readwrite("isy", &block_::isy)
      .def_readwrite("isz", &block_::isz)
      .def_readwrite("iS", &block_::iS)
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
      .def_readwrite("jS", &block_::jS)
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
      .def_readwrite("kS", &block_::kS)
      .def_readwrite("knx", &block_::knx)
      .def_readwrite("kny", &block_::kny)
      .def_readwrite("knz", &block_::knz)

      //----------------------------------------------------------------------------//
      //  Flow variables
      //----------------------------------------------------------------------------//
      // Conservative,primative variables
      .def_readwrite("Q", &block_::Q)
      .def_readwrite("q", &block_::q)
      .def_readwrite("dQ", &block_::dQ)

      // Spatial derivative of prim array
      .def_readwrite("dqdx", &block_::dqdx)
      .def_readwrite("dqdy", &block_::dqdy)
      .def_readwrite("dqdz", &block_::dqdz)

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

  py::class_<face_>(m, "face_", py::dynamic_attr())
      .def(py::init<>())

      .def_readwrite("_ng", &face_::_ng)
      .def_readwrite("_nface", &face_::_nface)

      .def_readwrite("qBcVals", &face_::qBcVals)
      .def_readwrite("QBcVals", &face_::QBcVals)
      .def_readwrite("sendBuffer_x", &face_::sendBuffer_x)
      .def_readwrite("sendBuffer_y", &face_::sendBuffer_y)
      .def_readwrite("sendBuffer_z", &face_::sendBuffer_z)
      .def_readwrite("sendBuffer_q", &face_::sendBuffer_q)
      .def_readwrite("sendBuffer_Q", &face_::sendBuffer_Q)
      .def_readwrite("sendBuffer_dqdx", &face_::sendBuffer_dqdx)
      .def_readwrite("sendBuffer_dqdy", &face_::sendBuffer_dqdy)
      .def_readwrite("sendBuffer_dqdz", &face_::sendBuffer_dqdz)
      .def_readwrite("sendBuffer_phi", &face_::sendBuffer_phi)

      .def_readwrite("recvBuffer_x", &face_::recvBuffer_x)
      .def_readwrite("recvBuffer_y", &face_::recvBuffer_y)
      .def_readwrite("recvBuffer_z", &face_::recvBuffer_z)
      .def_readwrite("recvBuffer_q", &face_::recvBuffer_q)
      .def_readwrite("recvBuffer_Q", &face_::recvBuffer_Q)
      .def_readwrite("recvBuffer_dqdx", &face_::recvBuffer_dqdx)
      .def_readwrite("recvBuffer_dqdy", &face_::recvBuffer_dqdy)
      .def_readwrite("recvBuffer_dqdz", &face_::recvBuffer_dqdz)
      .def_readwrite("recvBuffer_phi", &face_::recvBuffer_phi)

      .def_readwrite("tempRecvBuffer_x", &face_::tempRecvBuffer_x)
      .def_readwrite("tempRecvBuffer_y", &face_::tempRecvBuffer_y)
      .def_readwrite("tempRecvBuffer_z", &face_::tempRecvBuffer_z)
      .def_readwrite("tempRecvBuffer_q", &face_::tempRecvBuffer_q)
      .def_readwrite("tempRecvBuffer_Q", &face_::tempRecvBuffer_Q)
      .def_readwrite("tempRecvBuffer_dqdx", &face_::tempRecvBuffer_dqdx)
      .def_readwrite("tempRecvBuffer_dqdy", &face_::tempRecvBuffer_dqdy)
      .def_readwrite("tempRecvBuffer_dqdz", &face_::tempRecvBuffer_dqdz)
      .def_readwrite("tempRecvBuffer_phi", &face_::tempRecvBuffer_phi)

      .def_readwrite("periodicRotMatrixUp", &face_::periodicRotMatrixUp)
      .def_readwrite("periodicRotMatrixDown", &face_::periodicRotMatrixDown)

      .def_readwrite("intervalDt", &face_::intervalDt)
      .def_readwrite("cubicSplineAlphas", &face_::cubicSplineAlphas)
      .def_readwrite("intervalAlphas", &face_::intervalAlphas);

  static auto _atexit = []() {
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  };

  atexit(_atexit);
}

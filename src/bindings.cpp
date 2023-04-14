#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

//--------------------------------------------------------------------------------------//
//
//        The python module
//
//--------------------------------------------------------------------------------------//

void bindAdvFlux(nb::module_ &);
void bindBoundaryConditions(nb::module_ &);
void bindChemistry(nb::module_ &);
void bindDiffFlux(nb::module_ &);
void bindSubgrid(nb::module_ &);
void bindSwitches(nb::module_ &);
void bindThermo(nb::module_ &);
void bindTransport(nb::module_ &);
void bindTimeIntegration(nb::module_ &);
void bindUtils(nb::module_ &);
void bindKokkos(nb::module_ &);

NB_MODULE(compute, m) {
  m.doc() = "Module to expose compute units written in C++ with Kokkos";

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

  ////////////////////////////////////////////////////////////////////////////////
  ///////////////////  C++ Parent block_ class
  //////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  nb::class_<block_>(m, "block_", nb::dynamic_attr())
      .def(nb::init<>())

      .def_rw("nblki", &block_::nblki)

      .def_rw("ni", &block_::ni)
      .def_rw("nj", &block_::nj)
      .def_rw("nk", &block_::nk)
      .def_rw("ng", &block_::ng)

      .def_rw("ne", &block_::ne)
#ifdef NSCOMPILE
      .def_ro("ns", &block_::ns)
#endif

      //----------------------------------------------------------------------------//
      //  Primary grid node coordinates
      //----------------------------------------------------------------------------//
      .def_rw("x", &block_::x)
      .def_rw("y", &block_::y)
      .def_rw("z", &block_::z)

      //----------------------------------------------------------------------------//
      //  Primary metrics
      //----------------------------------------------------------------------------//
      // Cell Centers
      .def_rw("xc", &block_::xc)
      .def_rw("yc", &block_::yc)
      .def_rw("zc", &block_::zc)
      .def_rw("J", &block_::J)
      // Cell center metrics
      .def_rw("dEdx", &block_::dEdx)
      .def_rw("dEdy", &block_::dEdy)
      .def_rw("dEdz", &block_::dEdz)
      .def_rw("dNdx", &block_::dNdx)
      .def_rw("dNdy", &block_::dNdy)
      .def_rw("dNdz", &block_::dNdz)
      .def_rw("dCdx", &block_::dCdx)
      .def_rw("dCdy", &block_::dCdy)
      .def_rw("dCdz", &block_::dCdz)
      // i Face centers
      .def_rw("ixc", &block_::ixc)
      .def_rw("iyc", &block_::iyc)
      .def_rw("izc", &block_::izc)
      // i face area vector
      .def_rw("isx", &block_::isx)
      .def_rw("isy", &block_::isy)
      .def_rw("isz", &block_::isz)
      .def_rw("iS", &block_::iS)
      .def_rw("inx", &block_::inx)
      .def_rw("iny", &block_::iny)
      .def_rw("inz", &block_::inz)
      // j Face centers
      .def_rw("jxc", &block_::jxc)
      .def_rw("jyc", &block_::jyc)
      .def_rw("jzc", &block_::jzc)
      // j face area vector
      .def_rw("jsx", &block_::jsx)
      .def_rw("jsy", &block_::jsy)
      .def_rw("jsz", &block_::jsz)
      .def_rw("jS", &block_::jS)
      .def_rw("jnx", &block_::jnx)
      .def_rw("jny", &block_::jny)
      .def_rw("jnz", &block_::jnz)
      // k Face centers
      .def_rw("kxc", &block_::kxc)
      .def_rw("kyc", &block_::kyc)
      .def_rw("kzc", &block_::kzc)
      // k face area vector
      .def_rw("ksx", &block_::ksx)
      .def_rw("ksy", &block_::ksy)
      .def_rw("ksz", &block_::ksz)
      .def_rw("kS", &block_::kS)
      .def_rw("knx", &block_::knx)
      .def_rw("kny", &block_::kny)
      .def_rw("knz", &block_::knz)

      //----------------------------------------------------------------------------//
      //  Flow variables
      //----------------------------------------------------------------------------//
      // Conservative,primative variables
      .def_rw("Q", &block_::Q)
      .def_rw("q", &block_::q)
      .def_rw("dQ", &block_::dQ)

      // Spatial derivative of prim array
      .def_rw("dqdx", &block_::dqdx)
      .def_rw("dqdy", &block_::dqdy)
      .def_rw("dqdz", &block_::dqdz)

      // Thermo,transport variables
      .def_rw("qh", &block_::qh)
      .def_rw("qt", &block_::qt)

      // Chemistry
      .def_rw("omega", &block_::omega)

      // Time Integration Storage
      .def_rw("Q0", &block_::Q0)
      .def_rw("Q1", &block_::Q1)
      .def_rw("Q2", &block_::Q2)
      .def_rw("Q3", &block_::Q3)
      .def_rw("Qn", &block_::Qn)
      .def_rw("Qnm1", &block_::Qnm1)
      .def_rw("dtau", &block_::dtau)

      // Flux Arrays
      .def_rw("iF", &block_::iF)
      .def_rw("jF", &block_::jF)
      .def_rw("kF", &block_::kF)

      // Switch
      .def_rw("phi", &block_::phi);

  nb::class_<face_>(m, "face_", nb::dynamic_attr())
      .def(nb::init<>())

      .def_rw("_ng", &face_::_ng)
      .def_rw("_nface", &face_::_nface)

      .def_rw("qBcVals", &face_::qBcVals)
      .def_rw("QBcVals", &face_::QBcVals)
      .def_rw("sendBuffer_x", &face_::sendBuffer_x)
      .def_rw("sendBuffer_y", &face_::sendBuffer_y)
      .def_rw("sendBuffer_z", &face_::sendBuffer_z)
      .def_rw("sendBuffer_q", &face_::sendBuffer_q)
      .def_rw("sendBuffer_Q", &face_::sendBuffer_Q)
      .def_rw("sendBuffer_dqdx", &face_::sendBuffer_dqdx)
      .def_rw("sendBuffer_dqdy", &face_::sendBuffer_dqdy)
      .def_rw("sendBuffer_dqdz", &face_::sendBuffer_dqdz)
      .def_rw("sendBuffer_phi", &face_::sendBuffer_phi)

      .def_rw("recvBuffer_x", &face_::recvBuffer_x)
      .def_rw("recvBuffer_y", &face_::recvBuffer_y)
      .def_rw("recvBuffer_z", &face_::recvBuffer_z)
      .def_rw("recvBuffer_q", &face_::recvBuffer_q)
      .def_rw("recvBuffer_Q", &face_::recvBuffer_Q)
      .def_rw("recvBuffer_dqdx", &face_::recvBuffer_dqdx)
      .def_rw("recvBuffer_dqdy", &face_::recvBuffer_dqdy)
      .def_rw("recvBuffer_dqdz", &face_::recvBuffer_dqdz)
      .def_rw("recvBuffer_phi", &face_::recvBuffer_phi)

      .def_rw("tempRecvBuffer_x", &face_::tempRecvBuffer_x)
      .def_rw("tempRecvBuffer_y", &face_::tempRecvBuffer_y)
      .def_rw("tempRecvBuffer_z", &face_::tempRecvBuffer_z)
      .def_rw("tempRecvBuffer_q", &face_::tempRecvBuffer_q)
      .def_rw("tempRecvBuffer_Q", &face_::tempRecvBuffer_Q)
      .def_rw("tempRecvBuffer_dqdx", &face_::tempRecvBuffer_dqdx)
      .def_rw("tempRecvBuffer_dqdy", &face_::tempRecvBuffer_dqdy)
      .def_rw("tempRecvBuffer_dqdz", &face_::tempRecvBuffer_dqdz)
      .def_rw("tempRecvBuffer_phi", &face_::tempRecvBuffer_phi)

      .def_rw("periodicRotMatrixUp", &face_::periodicRotMatrixUp)
      .def_rw("periodicRotMatrixDown", &face_::periodicRotMatrixDown)

      .def_rw("intervalDt", &face_::intervalDt)
      .def_rw("cubicSplineAlphas", &face_::cubicSplineAlphas)
      .def_rw("intervalAlphas", &face_::intervalAlphas);

  static auto _atexit = []() {
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  };

  atexit(_atexit);
}

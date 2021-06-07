#ifndef __kokkos2peregrine_H__
#define __kokkos2peregrine_H__

#include "Kokkos_Core.hpp"

// Define the execution and storage space
#if defined(KOKKOS_ENABLE_CUDA_UVM)
using exec_space = Kokkos::Cuda;
using view_space = Kokkos::CudaUVMSpace;
static const std::string KokkosLocation = "CudaUVM";
#elif defined(KOKKOS_ENABLE_CUDA)
using exec_space = Kokkos::Cuda;
using view_space = Kokkos::CudaSpace;
static const std::string KokkosLocation = "Cuda";
#elif defined(KOKKOS_ENABLE_HIP)
using exec_space = Kokkos::Experimental::HIP;
using view_space = Kokkos::Experimental::HIPSpace;
static const std::string KokkosLocation = "HIP";
#elif defined(KOKKOS_ENABLE_OPENMP)
using exec_space = Kokkos::OpenMP;
using view_space = Kokkos::HostSpace;
static const std::string KokkosLocation = "OpenMP";
#elif defined(KOKKOS_ENABLE_DEAFULT)
using exec_space = Kokkos::DefaultHostExecutionSpace;
using view_space = Kokkos::HostSpace;
static const std::string KokkosLocation = "Default";
#endif

// define some shorthand for the Kokkos views and Range Policies
using oneDview   = Kokkos::View<double*,     Kokkos::LayoutRight, view_space>;
using twoDview   = Kokkos::View<double**,    Kokkos::LayoutRight, view_space>;
using threeDview = Kokkos::View<double***,   Kokkos::LayoutRight, view_space>;
using fourDview  = Kokkos::View<double****,  Kokkos::LayoutRight, view_space>;
using fiveDview  = Kokkos::View<double*****, Kokkos::LayoutRight, view_space>;
using MDRange1   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<1>>;
using MDRange2   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<2>>;
using MDRange3   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<3>>;
using MDRange4   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<4>>;
using MDRange5   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<5>>;

threeDview gen3Dview(std::string name, int ni, int nj, int nk) {
  if (!Kokkos::is_initialized()) {
    std::cerr << "[user-bindings]> Initializing Kokkos..." << std::endl;
    Kokkos::initialize();
  }
  threeDview _v(name, ni, nj, nk);
  MDRange3 _range({{0,0,0}},{{ni,nj,nk}});
  Kokkos::parallel_for("Gen3", _range, KOKKOS_LAMBDA(const int i,
                                                     const int j,
                                                     const int k) {
    _v(i,j,k) = 0.0;
  });
  return _v;
}

fourDview gen4Dview(std::string name, int ni, int nj, int nk, int nl) {
  if (!Kokkos::is_initialized()) {
    std::cerr << "[user-bindings]> Initializing Kokkos..." << std::endl;
    Kokkos::initialize();
  }
  fourDview _v(name, ni, nj, nk, nl);
  MDRange4 _range({{0,0,0,0}},{{ni,nj,nk,nl}});
  Kokkos::parallel_for("Gen4", _range, KOKKOS_LAMBDA(const int i,
                                                     const int j,
                                                     const int k,
                                                     const int l) {
    _v(i,j,k,l) = 0.0;
  });
  return _v;
}

#endif

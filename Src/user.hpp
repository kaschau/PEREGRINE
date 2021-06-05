
#include "Kokkos_Core.hpp"
#include <cstdint>

#if defined(KOKKOS_ENABLE_CUDA_UVM)
using exec_space = Kokkos::Cuda;
using view_space = Kokkos::CudaUVMSpace;
#elif defined(KOKKOS_ENABLE_CUDA)
using exec_space = Kokkos::Cuda;
using view_space = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using exec_space = Kokkos::Experimental::HIP;
using view_space = Kokkos::Experimental::HIPSpace;
#else
using exec_space = Kokkos::DefaultHostExecutionSpace;
//using exec_space = Kokkos::Serial;
using view_space = Kokkos::HostSpace;
#endif

using oneDview   = Kokkos::View<double*,    Kokkos::LayoutRight, view_space>;
using twoDview   = Kokkos::View<double**,   Kokkos::LayoutRight, view_space>;
using threeDview = Kokkos::View<double***,  Kokkos::LayoutRight, view_space>;
using fourDview  = Kokkos::View<double****, Kokkos::LayoutRight, view_space>;
using MDRange2   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<2>>;
using MDRange3   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<3>>;


struct block {
  int nblki;
  int nx,ny,nz;

  threeDview x,y,z;
};

void add(oneDview kkview, double n);
void add2(threeDview kkview, double n, int imin, int jmin, int kmin,
                                       int imax, int jmax, int kmax);
void add3(block b, double n);

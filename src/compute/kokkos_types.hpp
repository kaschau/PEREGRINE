#ifndef __kokkos_types_H__
#define __kokkos_types_H__

#include "Kokkos_Core.hpp"

// Define the execution and storage space
#if defined( KOKKOS_ENABLE_CUDA )
using exec_space = Kokkos::Cuda;
using view_space = Kokkos::CudaSpace;
using layout = Kokkos::LayoutLeft;
static const std::string KokkosLocation = "Cuda";
#elif defined( KOKKOS_ENABLE_HIP )
using exec_space = Kokkos::Experimental::HIP;
using view_space = Kokkos::Experimental::HIPSpace;
using layout = Kokkos::LayoutLeft;
static const std::string KokkosLocation = "HIP";
#elif defined( KOKKOS_ENABLE_OPENMPTARGET )
using exec_space = Kokkos::OpenMPTarget;
using view_space = Kokkos::OpenMPTargetSpace;
using layout = Kokkos::LayoutLeft;
#elif defined( KOKKOS_ENABLE_OPENMP )
using exec_space = Kokkos::OpenMP;
using view_space = Kokkos::HostSpace;
using layout = Kokkos::LayoutRight;
static const std::string KokkosLocation = "OpenMP";
#elif defined( KOKKOS_ENABLE_SERIAL )
using exec_space = Kokkos::Serial;
using view_space = Kokkos::HostSpace;
using layout = Kokkos::LayoutRight;
static const std::string KokkosLocation = "Serial";
#endif

// define some shorthand for the Kokkos views and Range Policies
using oneDview   = Kokkos::View<double*,     layout, view_space>;
using oneDsubview = Kokkos::View<double*,    Kokkos::LayoutStride, view_space>;
using twoDview   = Kokkos::View<double**,    layout, view_space>;
using twoDsubview = Kokkos::View<double**,   Kokkos::LayoutStride, view_space>;
using threeDview = Kokkos::View<double***,   layout, view_space>;
using threeDsubview = Kokkos::View<double***,   Kokkos::LayoutStride, view_space>;
using fourDview  = Kokkos::View<double****,  layout, view_space>;
using fiveDview  = Kokkos::View<double*****, layout, view_space>;
using MDRange1   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<1>>;
using MDRange2   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<2>>;
using MDRange3   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<3>>;
using MDRange4   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<4>>;
using MDRange5   = Kokkos::MDRangePolicy<exec_space,Kokkos::Rank<5>>;


// Always host stuff
using host_space = Kokkos::HostSpace;
using fourDviewHostsubview  = Kokkos::View<double****, Kokkos::LayoutStride, host_space>;
using fiveDviewHost  = Kokkos::View<double*****, Kokkos::LayoutRight, host_space>;
#endif

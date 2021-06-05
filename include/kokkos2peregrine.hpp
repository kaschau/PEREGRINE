
#include "Kokkos_Core.hpp"

// Define the execution and storage space
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
using view_space = Kokkos::HostSpace;
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
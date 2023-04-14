#ifndef __kokkosTypes_H__
#define __kokkosTypes_H__

#include "Kokkos_Core.hpp"

// Define the execution and storage space
#if defined(KOKKOS_ENABLE_CUDA)
using execSpace = Kokkos::Cuda;
using viewSpace = Kokkos::CudaSpace;
using layout = Kokkos::LayoutLeft;
#elif defined(KOKKOS_ENABLE_HIP)
using execSpace = Kokkos::Experimental::HIP;
using viewSpace = Kokkos::Experimental::HIPSpace;
using layout = Kokkos::LayoutLeft;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
using execSpace = Kokkos::OpenMPTarget;
using viewSpace = Kokkos::OpenMPTargetSpace;
using layout = Kokkos::LayoutLeft;
#elif defined(KOKKOS_ENABLE_OPENMP)
using execSpace = Kokkos::OpenMP;
using viewSpace = Kokkos::HostSpace;
using layout = Kokkos::LayoutRight;
#elif defined(KOKKOS_ENABLE_SERIAL)
using execSpace = Kokkos::Serial;
using viewSpace = Kokkos::HostSpace;
using layout = Kokkos::LayoutRight;
#endif

using defaultViewHooks = Kokkos::Experimental::DefaultViewHooks;
// define some shorthand for the Kokkos views and Range Policies
using oneDview = Kokkos::View<double *, layout, viewSpace, defaultViewHooks>;
using oneDsubview =
    Kokkos::View<double *, Kokkos::LayoutStride, viewSpace, defaultViewHooks>;
using twoDview = Kokkos::View<double **, layout, viewSpace, defaultViewHooks>;
using twoDviewInt = Kokkos::View<int **, layout, viewSpace, defaultViewHooks>;
using twoDsubview =
    Kokkos::View<double **, Kokkos::LayoutStride, viewSpace, defaultViewHooks>;
using threeDview =
    Kokkos::View<double ***, layout, viewSpace, defaultViewHooks>;
using threeDsubview =
    Kokkos::View<double ***, Kokkos::LayoutStride, viewSpace, defaultViewHooks>;
using fourDview =
    Kokkos::View<double ****, layout, viewSpace, defaultViewHooks>;
using fiveDview =
    Kokkos::View<double *****, layout, viewSpace, defaultViewHooks>;
using MDRange1 = Kokkos::MDRangePolicy<execSpace, Kokkos::Rank<1>>;
using MDRange2 = Kokkos::MDRangePolicy<execSpace, Kokkos::Rank<2>>;
using MDRange3 = Kokkos::MDRangePolicy<execSpace, Kokkos::Rank<3>>;
using MDRange4 = Kokkos::MDRangePolicy<execSpace, Kokkos::Rank<4>>;
using MDRange5 = Kokkos::MDRangePolicy<execSpace, Kokkos::Rank<5>>;

// Always host stuff
using hostSpace = Kokkos::HostSpace;
using fourDviewHostsubview =
    Kokkos::View<double ****, Kokkos::LayoutStride, hostSpace>;
using fiveDviewHost = Kokkos::View<double *****, Kokkos::LayoutRight, hostSpace,
                                   defaultViewHooks>;
#endif

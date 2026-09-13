#ifndef __kokkosTypes_H__
#define __kokkosTypes_H__

#include <Kokkos_Core.hpp>

// where kernels run, and the memory and layout Kokkos picks for it
using execSpace = Kokkos::DefaultExecutionSpace;
using viewSpace = execSpace::memory_space;
using layout = execSpace::array_layout;
using hostSpace = Kokkos::HostSpace;

// a kernel's view of an array Python owns
template <class T>
using unmanaged =
    Kokkos::View<T, layout, viewSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
// by rank, what a kernel reads and what it writes
using in1 = unmanaged<const double *>;
using in2 = unmanaged<const double **>;
using in3 = unmanaged<const double ***>;
using in4 = unmanaged<const double ****>;
using in5 = unmanaged<const double *****>;
using out1 = unmanaged<double *>;
using out2 = unmanaged<double **>;
using out3 = unmanaged<double ***>;
using out4 = unmanaged<double ****>;
using out5 = unmanaged<double *****>;
// a slice of one: strided whichever index is fixed
template <class T>
using strided = Kokkos::View<T, Kokkos::LayoutStride, viewSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// scratch a kernel allocates for itself
using twoDview = Kokkos::View<double **, layout, viewSpace>;
using twoDviewInt = Kokkos::View<int **, layout, viewSpace>;
using threeDview = Kokkos::View<double ***, layout, viewSpace>;

using MDRange2 = Kokkos::MDRangePolicy<execSpace, Kokkos::Rank<2>>;
using MDRange3 = Kokkos::MDRangePolicy<execSpace, Kokkos::Rank<3>>;
using MDRange4 = Kokkos::MDRangePolicy<execSpace, Kokkos::Rank<4>>;

#endif

// Packing a face's planes into a buffer turned onto the neighbor's frame, and
// back.
#ifndef __faceBuffers_H__
#define __faceBuffers_H__

#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

// A face's halo trade. The send buffer is packed the way the neighbor reads
// it -- the plane is turned on the way out rather than after it lands -- and
// the receive buffer lands as it is. `slices` lists the plane index of each
// layer along the face's normal; a buffer's leading extent is the layer.

template <class View, class Buffer>
static void pack(const View &view, const Buffer &buffer, int nface,
                 const int *slices, int nLayer, bool transpose, bool flip0,
                 bool flip1) {
  for (int g = 0; g < nLayer; g++) {
    auto viewSlice = getFaceSlice(view, nface, slices[g]);
    auto bufferSlice = getFaceSlice(buffer, 1, g);
    const int na = bufferSlice.extent(0);
    const int nb = bufferSlice.extent(1);
    if constexpr (View::rank == 4) {
      MDRange3 range({0, 0, 0}, {na, nb, (int)bufferSlice.extent(2)});
      Kokkos::parallel_for(
          "extract send buffer", range,
          KOKKOS_LAMBDA(const int a, const int b, const int l) {
            const int aa = flip0 ? na - 1 - a : a;
            const int bb = flip1 ? nb - 1 - b : b;
            bufferSlice(a, b, l) =
                transpose ? viewSlice(bb, aa, l) : viewSlice(aa, bb, l);
          });
    } else {
      MDRange4 range({0, 0, 0, 0}, {na, nb, (int)bufferSlice.extent(2),
                                    (int)bufferSlice.extent(3)});
      Kokkos::parallel_for(
          "extract send buffer", range,
          KOKKOS_LAMBDA(const int a, const int b, const int l, const int m) {
            const int aa = flip0 ? na - 1 - a : a;
            const int bb = flip1 ? nb - 1 - b : b;
            bufferSlice(a, b, l, m) =
                transpose ? viewSlice(bb, aa, l, m) : viewSlice(aa, bb, l, m);
          });
    }
  }
}

template <class View, class Buffer>
static void unpack(const View &view, const Buffer &buffer, int nface,
                   const int *slices, int nLayer) {
  for (int g = 0; g < nLayer; g++) {
    auto viewSlice = getFaceSlice(view, nface, slices[g]);
    auto bufferSlice = getFaceSlice(buffer, 1, g);
    Kokkos::deep_copy(viewSlice, bufferSlice);
  }
}

#endif

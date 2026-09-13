// Packing a face's planes into a buffer turned onto the neighbor's frame, and
// back.
#ifndef __faceBuffers_H__
#define __faceBuffers_H__

#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

// A face's halo trade. Layer 0 of a buffer is the plane nearest the face on
// both sides, so what one block sends from its interior lands in its
// neighbor's halo layer for layer whatever side either face is on. The send
// buffer is packed the way the neighbor reads it -- the plane is turned on
// the way out rather than after it lands -- and the receive buffer lands as
// it is.

template <class View, class Buffer>
static void pack(const View &view, const Buffer &buffer, int nface, int nLayer,
                 int skip, bool transpose, bool flip0, bool flip1) {
  auto out = interior(view, nface, nLayer, skip);
  const int na = buffer.extent(1);
  const int nb = buffer.extent(2);
  if constexpr (View::rank == 4) {
    MDRange4 range({0, 0, 0, 0}, {nLayer, na, nb, (int)buffer.extent(3)});
    Kokkos::parallel_for(
        "extract send buffer", range,
        KOKKOS_LAMBDA(const int g, const int a, const int b, const int l) {
          const int aa = flip0 ? na - 1 - a : a;
          const int bb = flip1 ? nb - 1 - b : b;
          buffer(g, a, b, l) =
              transpose ? out(g, bb, aa, l) : out(g, aa, bb, l);
        });
  } else {
    MDRange5 range({0, 0, 0, 0, 0}, {nLayer, na, nb, (int)buffer.extent(3),
                                     (int)buffer.extent(4)});
    Kokkos::parallel_for(
        "extract send buffer", range,
        KOKKOS_LAMBDA(const int g, const int a, const int b, const int l,
                      const int m) {
          const int aa = flip0 ? na - 1 - a : a;
          const int bb = flip1 ? nb - 1 - b : b;
          buffer(g, a, b, l, m) =
              transpose ? out(g, bb, aa, l, m) : out(g, aa, bb, l, m);
        });
  }
}

template <class View, class Buffer>
static void unpack(const View &view, const Buffer &buffer, int nface,
                   int nLayer) {
  auto in = halo(view, nface, nLayer);
  const int na = buffer.extent(1);
  const int nb = buffer.extent(2);
  if constexpr (View::rank == 4) {
    MDRange4 range({0, 0, 0, 0}, {nLayer, na, nb, (int)buffer.extent(3)});
    Kokkos::parallel_for(
        "place recv buffer", range,
        KOKKOS_LAMBDA(const int g, const int a, const int b, const int l) {
          in(g, a, b, l) = buffer(g, a, b, l);
        });
  } else {
    MDRange5 range({0, 0, 0, 0, 0}, {nLayer, na, nb, (int)buffer.extent(3),
                                     (int)buffer.extent(4)});
    Kokkos::parallel_for(
        "place recv buffer", range,
        KOKKOS_LAMBDA(const int g, const int a, const int b, const int l,
                      const int m) {
          in(g, a, b, l, m) = buffer(g, a, b, l, m);
        });
  }
}

#endif

// Packing every trading face's planes into its buffer turned onto the
// neighbor's frame, and back, each in one launch.
#ifndef __faceBuffers_H__
#define __faceBuffers_H__

#include "kernel.hpp"

// One block face's trade in a halo exchange. Layer 0 of a buffer is the
// plane nearest the face on
// both sides, so what one block sends from its interior lands in its
// neighbor's halo layer for layer whatever side either face is on. The send
// buffer is packed the way the neighbor reads it -- the plane is turned on
// the way out rather than after it lands -- and the receive buffer lands as
// it is. A thread stands on one plane cell of one layer and does every
// component of it; a variable's faces are all of one ndim.

// what python hands a send: the block array, the buffer, and how the
// neighbor's frame differs (a node array's planes start past the face)
struct haloSend {
  haloIn view;
  bufferOut buffer;
  perEntry<int> skip, transpose, flip0, flip1;
  int ndim;
};
template <int R> struct packing {
  haloIn view;
  bufferOut buffer;
  perEntry<int> skip, transpose, flip0, flip1;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int g = view.p.g, a = view.p.i, b = view.p.j;
    const int na = buffer.extent(1), nb = buffer.extent(2);
    const int nl = buffer.extent(3), nm = R == 5 ? buffer.extent(4) : 1;
    // the interior cell this buffer cell takes, in the neighbor's frame
    const int aa = flip0() ? na - 1 - a : a;
    const int bb = flip1() ? nb - 1 - b : b;
    const auto src = transpose() ? view.on(bb, aa) : view.on(aa, bb);
    const int layer = g + skip();
    for (int m = 0; m < nm; m++)
      for (int l = 0; l < nl; l++) {
        if constexpr (R == 4)
          buffer(g, a, b, l) = src.at(layer, l);
        else
          buffer(g, a, b, l, m) = src.at(layer, l, m);
      }
  }
};

struct haloRecv {
  haloOut view;
  bufferIn buffer;
  int ndim;
};
template <int R> struct unpacking {
  haloOut view;
  bufferIn buffer;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int g = view.p.g, a = view.p.i, b = view.p.j;
    const int nl = buffer.extent(3), nm = R == 5 ? buffer.extent(4) : 1;
    for (int m = 0; m < nm; m++)
      for (int l = 0; l < nl; l++) {
        if constexpr (R == 4)
          view.L(l) = buffer(g, a, b, l);
        else
          view.L(l, m) = buffer(g, a, b, l, m);
      }
  }
};

#endif

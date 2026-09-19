// Packing every trading face's planes into its buffer turned onto the
// neighbor's frame, and back, each in one launch; and a trade met on this
// rank filled directly, with no buffer between; and the halo behind a
// rotational periodic face turned by the face's rotation once it has
// landed.
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

// what python hands a send: the block array, the buffer, how the neighbor's
// frame differs, and how far past the face the planes start (one for a
// node array, whose face plane both sides hold)
struct haloSend {
  haloIn view;
  bufferOut buffer;
  perEntry<int> transpose, flip0, flip1;
  int skip, ndim;
};
template <int R> struct packing {
  haloIn view;
  bufferOut buffer;
  perEntry<int> transpose, flip0, flip1;
  int skip;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int g = view.p.g, a = view.p.i, b = view.p.j;
    const int na = buffer.extent(1), nb = buffer.extent(2);
    const int nl = buffer.extent(3), nm = R == 5 ? buffer.extent(4) : 1;
    // the interior cell this buffer cell takes, in the neighbor's frame
    const int aa = flip0() ? na - 1 - a : a;
    const int bb = flip1() ? nb - 1 - b : b;
    const auto src = transpose() ? view.on(bb, aa) : view.on(aa, bb);
    const int layer = g + skip;
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

// what python hands a turn: the block array, the face's rotation, and
// which of the array's components are vectors: for an array of one index
// per cell the component a 3-vector starts at, for one of two the last
// index of every slot
struct haloTurn {
  haloInOut view;
  plainIn rotation;
  int vectors, ndim;
};
// The vectors of a halo cell behind a rotational periodic face, turned by
// the face's rotation once the halo has landed: a periodic is the
// topology's, and the exchange is where it acts.
template <int R> struct turning {
  haloInOut view;
  plainIn rotation;
  int vectors;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const auto turn = [&](auto at) {
      fpdtype v[3];
      for (int m = 0; m < 3; m++)
        v[m] = at(m);
      for (int r = 0; r < 3; r++)
        at(r) = rotation(r, 0) * v[0] + rotation(r, 1) * v[1] +
                rotation(r, 2) * v[2];
    };
    if constexpr (R == 4) {
      turn([&](const int m) -> fpdtype & { return view.L(vectors + m); });
    } else {
      const int nl = view.extent(3);
      for (int l = 0; l < nl; l++)
        turn([&](const int m) -> fpdtype & { return view.L(l, m); });
    }
  }
};

// what python hands a direct fill: our block array and the same array of the
// block across (the face's partner column), the partner's face and how it
// would have turned its plane for us, and how far past the face the
// planes start
struct directHaloFill {
  haloOut view;
  haloIn theirs;
  perEntry<int> partnerNface, partnerTranspose, partnerFlip0, partnerFlip1;
  int skip, ndim;
};
// A trade met on this rank, with no buffer between: a thread stands on one
// of our halo cells and reads the interior cell across that the partner
// would have packed for it, at the position its pack would have turned
// ours to
template <int R> struct directFilling {
  haloOut view;
  haloIn theirs;
  perEntry<int> partnerNface, partnerTranspose, partnerFlip0, partnerFlip1;
  int skip;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int g = view.p.g, a = view.p.i, b = view.p.j;
    // our face plane's extents, what the buffer's would have been
    int n[2], d = 0;
    for (int k = 0; k < 3; k++)
      if (k != view.axis())
        n[d++] = view.extent(k) - 2 * ng;
    const int aa = partnerFlip0() ? n[0] - 1 - a : a;
    const int bb = partnerFlip1() ? n[1] - 1 - b : b;
    haloIn src = theirs;
    src.p.nface = partnerNface();
    src.p.i = partnerTranspose() ? bb : aa;
    src.p.j = partnerTranspose() ? aa : bb;
    const int layer = g + skip;
    const int nl = view.extent(3), nm = R == 5 ? view.extent(4) : 1;
    for (int m = 0; m < nm; m++)
      for (int l = 0; l < nl; l++) {
        if constexpr (R == 4)
          view.L(l) = src.at(layer, l);
        else
          view.L(l, m) = src.at(layer, l, m);
      }
  }
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

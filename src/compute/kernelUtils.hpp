// What the kernels share: the loop range Python states, a face's normal,
// and the planes of any array either side of a face.
#ifndef __kernelUtils_H__
#define __kernelUtils_H__

#include "abi.hpp"
#include <utility>

// the case's constants, baked in by the jit
#if !defined(NS) || !defined(NE) || !defined(NG)
#error "a kernel is compiled for one case: NS, NE and NG come from the jit"
#endif
constexpr int ns = NS, ne = NE, ng = NG;

inline MDRange3 range3(const pgRange &r) {
  return MDRange3({r.i0, r.j0, r.k0}, {r.i1, r.j1, r.k1});
}

// a face's area and unit normal, from its area vector
KOKKOS_INLINE_FUNCTION
void faceNormal(const double &sx, const double &sy, const double &sz, double &S,
                double &nx, double &ny, double &nz) {
  // a degenerate face is floored, we divide by this
  S = sqrt(sx * sx + sy * sy + sz * sz);
  double Sinv = 1.0 / S;
  nx = sx * Sinv;
  ny = sy * Sinv;
  nz = sz * Sinv;
}

// a kernel with a wider stencil says so; the jit sizes the halo to the widest
#define PG_STENCIL(n)                                                          \
  static_assert(NG >= (n), "this kernel needs " #n " halo layers")

// A stack of planes of a block array, as a face walks them: the layer first,
// then the two in-plane indices, then the components. Layer 0 is the plane
// nearest the face and the layer stride carries the direction, so a halo runs
// outward and an interior inward with the same index.
template <class T, int R> struct planes {
  T *data;
  long stride[R];
  int extent[R];

  template <class... I> KOKKOS_INLINE_FUNCTION T &operator()(I... index) const {
    const long at[] = {static_cast<long>(index)...};
    long offset = 0;
    for (int d = 0; d < R; d++)
      offset += at[d] * stride[d];
    return data[offset];
  }
};

// the axis a face is normal to, and whether it is the low end of it
KOKKOS_INLINE_FUNCTION int faceAxis(const int nface) { return (nface - 1) / 2; }
KOKKOS_INLINE_FUNCTION bool faceLow(const int nface) { return nface % 2 == 1; }

// `count` planes of `view` from the one at `start`, stepping `step` along
// the face's axis; the in-plane axes keep their order
template <class View>
auto stackAlong(const View &view, const int nface, const int start,
                const int step, const int count) {
  constexpr int rank = View::rank;
  const int axis = faceAxis(nface);
  planes<typename View::value_type, rank> p;
  p.data = view.data() + start * view.stride(axis);
  p.stride[0] = step * static_cast<long>(view.stride(axis));
  p.extent[0] = count;
  int d = 1;
  for (int a = 0; a < rank; a++) {
    if (a == axis)
      continue;
    p.stride[d] = view.stride(a);
    p.extent[d] = view.extent(a);
    d++;
  }
  return p;
}

// the halo layers of an array, outward from the face
template <class View>
auto halo(const View &view, const int nface, const int count) {
  const int n = view.extent(faceAxis(nface));
  const bool low = faceLow(nface);
  return stackAlong(view, nface, low ? ng - 1 : n - ng, low ? -1 : 1, count);
}
// the interior layers of an array, inward from the face; a node array's
// start past the face plane itself, which both sides hold
template <class View>
auto interior(const View &view, const int nface, const int count,
              const int skip = 0) {
  const int n = view.extent(faceAxis(nface));
  const bool low = faceLow(nface);
  return stackAlong(view, nface, low ? ng + skip : n - ng - 1 - skip,
                    low ? 1 : -1, count);
}

#endif

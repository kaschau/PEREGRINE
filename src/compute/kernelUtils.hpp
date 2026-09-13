// What the kernels share: the loop range Python states, and a face's normal,
// its neighbouring cells and a slice of any view along it.
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

// T with R pointers on it: the data type of a rank-R view
template <class T, int R> struct pointers {
  using type = typename pointers<T, R - 1>::type *;
};
template <class T> struct pointers<T, 0> {
  using type = T;
};

// one subview argument: the fixed index on the face's axis, ALL elsewhere
template <bool onAxis> auto sliceArg(const int slice) {
  if constexpr (onAxis)
    return slice;
  else
    return Kokkos::ALL;
}
template <int axis, class View, std::size_t... dim>
auto sliceAlong(const View &view, const int slice,
                std::index_sequence<dim...>) {
  return Kokkos::subview(view, sliceArg<dim == axis>(slice)...);
}

// any view at one index along a face's normal, one rank lower
template <class View>
auto getFaceSlice(const View &view, const int nface, const int slice) {
  using out = strided<
      typename pointers<typename View::value_type, View::rank - 1>::type>;
  constexpr auto dims = std::make_index_sequence<View::rank>{};
  if (nface <= 2)
    return out(sliceAlong<0>(view, slice, dims));
  if (nface <= 4)
    return out(sliceAlong<1>(view, slice, dims));
  return out(sliceAlong<2>(view, slice, dims));
}

#endif

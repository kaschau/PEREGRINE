// The one header a kernel source includes: the two declarations a kernel
// makes (its stencil, its range), a face's normal, and through it the
// arrays and the launch. faces.hpp sits on this.
#ifndef __kernel_H__
#define __kernel_H__

#include "launch.hpp"
#include <math.h>

// In single, a two-argument math call on a fpdtype and a double literal,
// fmax(x, 1.0), is in fpdtype: otherwise the library's promoted template is
// chosen, which on nvcc is a host function, and a device body that calls
// one is silently dropped. A call already on two reals is the builtin; in
// double there is no such pair.
#ifdef PG_SINGLE
#define PG_FPDTYPE_PAIR(name)                                                  \
  KOKKOS_INLINE_FUNCTION fpdtype name(fpdtype a, double b) {                   \
    return ::name(a, fpdtype(b));                                              \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION fpdtype name(double a, fpdtype b) {                   \
    return ::name(fpdtype(a), b);                                              \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION fpdtype name(fpdtype a, int b) {                      \
    return ::name(a, fpdtype(b));                                              \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION fpdtype name(int a, fpdtype b) {                      \
    return ::name(fpdtype(a), b);                                              \
  }
PG_FPDTYPE_PAIR(fmin)
PG_FPDTYPE_PAIR(fmax)
PG_FPDTYPE_PAIR(pow)
#undef PG_FPDTYPE_PAIR
#endif

// a kernel with a wider stencil says so; the jit sizes the halo to the widest
#define PG_STENCIL(n)                                                          \
  static_assert(NG >= (n), "this kernel needs " #n " halo layers")
// a kernel declares the kind of item it runs over, one per tiling it
// takes; which cells is the launch's to say. PG_RANGE(cellCenters[,
// components = ne]) a cell of the block, or a cell and one of its
// components; PG_RANGE(cellFaces) a cell face of the kernel's direction;
// PG_RANGE(bufferPlanes) a plane cell of a trade's buffer; PG_RANGE(elements,
// components = ne) an element of an ne-wide array's allocation, flat
#define PG_RANGE(...)

#endif

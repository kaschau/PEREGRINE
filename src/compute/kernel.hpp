// The one header a kernel source includes: the two declarations a kernel
// makes (its stencil, its range), a face's normal, and through it the
// arrays and the launch. faces.hpp sits on this.
#ifndef __kernel_H__
#define __kernel_H__

#include "launch.hpp"
#include <math.h>

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
// a kernel declares the cells it does, one range per tiling it takes;
// Python reads the declaration and tiles the range over every entry
#define PG_RANGE(...)

#endif

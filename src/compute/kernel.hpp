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
// a kernel declares what it runs over, one range per tiling it takes, as a
// kind and its parameters; Python builds the range and tiles it over every
// entry: PG_RANGE(cellCenters[, halo = 1 | ng][, components = 3 | ne]) the
// interior and as much halo as it says, PG_RANGE(cellFaces) the faces of the
// kernel's direction, PG_RANGE(blockFacePlanes, layers = ng) a block face's
// planes, PG_RANGE(bufferPlanes) a trade's buffer
#define PG_RANGE(...)

#endif

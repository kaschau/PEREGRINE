// A cell face's normal: its area and unit normal from its area vector, and
// the velocity along that normal from what a state holds.
#ifndef __normal_H__
#define __normal_H__

#include "abi.hpp"

// a face's area and unit normal, from its area vector
KOKKOS_INLINE_FUNCTION
void faceNormal(const fpdtype &sx, const fpdtype &sy, const fpdtype &sz,
                fpdtype &S, fpdtype &nx, fpdtype &ny, fpdtype &nz) {
  // a degenerate face is floored, we divide by this
  S = sqrt(sx * sx + sy * sy + sz * sz);
  fpdtype Sinv = 1.0 / S;
  nx = sx * Sinv;
  ny = sy * Sinv;
  nz = sz * Sinv;
}

// the velocity along a unit normal, from the density and the momentum
KOKKOS_INLINE_FUNCTION
fpdtype normalVelocity(const fpdtype rho, const fpdtype rhou,
                       const fpdtype rhov, const fpdtype rhow, const fpdtype nx,
                       const fpdtype ny, const fpdtype nz) {
  return (nx * rhou + ny * rhov + nz * rhow) / rho;
}

#endif

#ifndef __adiabaticSlipWall_H__
#define __adiabaticSlipWall_H__

#include "kernel.hpp"

namespace adiabaticSlipWall {

struct euler {
  haloInOut q;
  blockFaceIn S;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double area, nx, ny, nz;
    faceNormal(S(0), S(1), S(2), area, nx, ny, nz);

    // match pressure
    q.L(0) = q.R(0);

    // mirror velo on wall
    double uDotn = q.R(1) * nx + q.R(2) * ny + q.R(3) * nz;
    q.L(1) = q.R(1) - 2.0 * uDotn * nx;
    q.L(2) = q.R(2) - 2.0 * uDotn * ny;
    q.L(3) = q.R(3) - 2.0 * uDotn * nz;

    // match temperature
    q.L(4) = q.R(4);
    // match species
    for (int n = 5; n < ne; n++) {
      q.L(n) = q.R(n);
    }
  }
};

struct postDqDxyz {
  haloInOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    for (int l = 0; l < ne; l++) {
      // negate all gradients
      for (int d = 0; d < 3; d++) {
        grads.L(l, d) = -grads.R(l, d);
      }
    }
  }
};

} // namespace adiabaticSlipWall

#endif

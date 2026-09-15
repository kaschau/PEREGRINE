#ifndef __isoTSlipWall_H__
#define __isoTSlipWall_H__

#include "kernel.hpp"

namespace isoTSlipWall {

struct euler {
  faceInOut q;
  faceIn S, qBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double area, nx, ny, nz;
    faceNormal(S.R(0), S.R(1), S.R(2), area, nx, ny, nz);

    // match pressure
    q.L(0) = q.R(0);

    // flip velo on wall
    double uDotn = q.R(1) * nx + q.R(2) * ny + q.R(3) * nz;
    q.L(1) = q.R(1) - 2.0 * uDotn * nx;
    q.L(2) = q.R(2) - 2.0 * uDotn * ny;
    q.L(3) = q.R(3) - 2.0 * uDotn * nz;

    // set temperature
    q.L(4) = qBcVals.here(4);
    // match species
    for (int n = 5; n < ne; n++) {
      q.L(n) = q.R(n);
    }
  }
};

struct postDqDxyz {
  faceInOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // negate velocity gradients
    grads.L(0, 0) = -grads.R(0, 0);
    grads.L(1, 0) = -grads.R(1, 0);
    grads.L(2, 0) = -grads.R(2, 0);
    grads.L(3, 0) = -grads.R(3, 0);

    grads.L(0, 1) = -grads.R(0, 1);
    grads.L(1, 1) = -grads.R(1, 1);
    grads.L(2, 1) = -grads.R(2, 1);
    grads.L(3, 1) = -grads.R(3, 1);

    grads.L(0, 2) = -grads.R(0, 2);
    grads.L(1, 2) = -grads.R(1, 2);
    grads.L(2, 2) = -grads.R(2, 2);
    grads.L(3, 2) = -grads.R(3, 2);

    // neumann temp gradients
    grads.L(4, 0) = grads.R(4, 0);
    grads.L(4, 1) = grads.R(4, 1);
    grads.L(4, 2) = grads.R(4, 2);

    // negate species gradient (so gradient evaluates
    // to zero on wall)
    for (int n = 5; n < ne; n++) {
      grads.L(n, 0) = -grads.R(n, 0);
      grads.L(n, 1) = -grads.R(n, 1);
      grads.L(n, 2) = -grads.R(n, 2);
    }
  }
};

} // namespace isoTSlipWall

#endif

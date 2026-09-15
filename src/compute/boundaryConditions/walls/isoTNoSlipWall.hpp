#ifndef __isoTNoSlipWall_H__
#define __isoTNoSlipWall_H__

#include "kernel.hpp"

namespace isoTNoSlipWall {

struct euler {
  haloInOut q;
  blockFaceIn S, qBcVals;
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

    // set temperature
    q.L(4) = qBcVals(4);
    // match species
    for (int n = 5; n < ne; n++) {
      q.L(n) = q.R(n);
    }
  }
};

struct preDqDxyz {
  haloInOut q;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // flip velo on wall
    q.L(1) = -q.R(1);
    q.L(2) = -q.R(2);
    q.L(3) = -q.R(3);
  }
};

struct postDqDxyz {
  haloInOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // negate pressure,  neumann velocity,temperature
    // gradients
    grads.L(0, 0) = -grads.R(0, 0);
    grads.L(1, 0) = grads.R(1, 0);
    grads.L(2, 0) = grads.R(2, 0);
    grads.L(3, 0) = grads.R(3, 0);
    grads.L(4, 0) = grads.R(4, 0);

    grads.L(0, 1) = -grads.R(0, 1);
    grads.L(1, 1) = grads.R(1, 1);
    grads.L(2, 1) = grads.R(2, 1);
    grads.L(3, 1) = grads.R(3, 1);
    grads.L(4, 1) = grads.R(4, 1);

    grads.L(0, 2) = -grads.R(0, 2);
    grads.L(1, 2) = grads.R(1, 2);
    grads.L(2, 2) = grads.R(2, 2);
    grads.L(3, 2) = grads.R(3, 2);
    grads.L(4, 2) = grads.R(4, 2);

    // negatespecies gradient (so gradient evaluates
    // to zero on wall)
    for (int n = 5; n < ne; n++) {
      grads.L(n, 0) = -grads.R(n, 0);
      grads.L(n, 1) = -grads.R(n, 1);
      grads.L(n, 2) = -grads.R(n, 2);
    }
  }
};

} // namespace isoTNoSlipWall

#endif

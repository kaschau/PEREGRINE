#ifndef __supersonicExit_H__
#define __supersonicExit_H__

#include "kernel.hpp"

namespace supersonicExit {

struct euler {
  faceInOut q;
  faceIn S;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const double dplus = S.outward();

    double area, nx, ny, nz;
    faceNormal(S.R(0), S.R(1), S.R(2), area, nx, ny, nz);

    // extrapolate, each halo layer mirrored about the first interior cell:
    // pressure (kept positive, and the wave exiting)
    q.L(0) = fmin(fmax(0.0, 2.0 * q.R(0) - q.at(q.p.g + 1, 0)), q.R(0));

    // velocity, unless reverse flow detected
    double uDotn = (q.R(1) * nx + q.R(2) * ny + q.R(3) * nz) * dplus;
    if (uDotn > 0.0) {
      for (int l = 1; l <= 3; l++) {
        q.L(l) = 2.0 * q.R(l) - q.at(q.p.g + 1, l);
      }
    } else {
      // flip velocity on face (like slip wall)
      q.L(1) = q.R(1) - 2.0 * uDotn * nx * dplus;
      q.L(2) = q.R(2) - 2.0 * uDotn * ny * dplus;
      q.L(3) = q.R(3) - 2.0 * uDotn * nz * dplus;
    }

    // temperature (kept positive)
    q.L(4) = fmax(0.0, 2.0 * q.R(4) - q.at(q.p.g + 1, 4));
    // species
    for (int l = 5; l < ne; l++) {
      q.L(l) = fmax(0.0, fmin(1.0, 2.0 * q.R(l) - q.at(q.p.g + 1, l)));
    }
  }
};

struct postDqDxyz {
  faceInOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    for (int l = 0; l < ne; l++) {
      // neumann all gradients
      for (int d = 0; d < 3; d++) {
        grads.L(l, d) = grads.R(l, d);
      }
    }
  }
};

} // namespace supersonicExit

#endif

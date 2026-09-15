#ifndef __periodicRot_H__
#define __periodicRot_H__

#include "kernel.hpp"

namespace periodicRot {

struct euler {
  haloInOut q, Q;
  recordIn rot;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // turn the velocity vector of this halo cell onto this face
    const double u = q.L(1);
    const double v = q.L(2);
    const double w = q.L(3);
    const double tempU = rot(0, 0) * u + rot(0, 1) * v + rot(0, 2) * w;
    const double tempV = rot(1, 0) * u + rot(1, 1) * v + rot(1, 2) * w;
    const double tempW = rot(2, 0) * u + rot(2, 1) * v + rot(2, 2) * w;

    // Update velocity
    q.L(1) = tempU;
    q.L(2) = tempV;
    q.L(3) = tempW;

    // Update momentum
    Q.L(1) = tempU * Q.L(0);
    Q.L(2) = tempV * Q.L(0);
    Q.L(3) = tempW * Q.L(0);
  }
};

struct postDqDxyz {
  haloInOut grads;
  recordIn rot;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    for (int l = 0; l < ne; l++) {
      // turn the gradient vectors onto this face
      double grad[3] = {grads.L(l, 0), grads.L(l, 1), grads.L(l, 2)};
      for (int r = 0; r < 3; r++) {
        double turned = 0.0;
        for (int c = 0; c < 3; c++) {
          turned += rot(r, c) * grad[c];
        }
        grads.L(l, r) = turned;
      }
    }
  }
};

} // namespace periodicRot

#endif

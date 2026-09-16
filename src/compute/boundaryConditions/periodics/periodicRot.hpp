#ifndef __periodicRot_H__
#define __periodicRot_H__

#include "kernel.hpp"

namespace periodicRot {

struct euler {
  haloInOut Q;
  recordIn rot;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // the halo came from the partner; its momentum turns with the face
    const double rhou = Q.L(1);
    const double rhov = Q.L(2);
    const double rhow = Q.L(3);
    Q.L(1) = rot(0, 0) * rhou + rot(0, 1) * rhov + rot(0, 2) * rhow;
    Q.L(2) = rot(1, 0) * rhou + rot(1, 1) * rhov + rot(1, 2) * rhow;
    Q.L(3) = rot(2, 0) * rhou + rot(2, 1) * rhov + rot(2, 2) * rhow;
  }
};

struct postDqDxyz {
  haloInOut grads;
  recordIn rot;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // every gradient turns with the face
    for (int l = 0; l < ne - 1; l++) {
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

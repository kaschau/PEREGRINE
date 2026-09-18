#ifndef __periodicRot_H__
#define __periodicRot_H__

#include "kernel.hpp"

namespace periodicRot {

struct euler {
  haloInOut Q;
  plainIn rotation;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // the halo came from the partner; its momentum turns with the face
    const fpdtype rhou = Q.L(1);
    const fpdtype rhov = Q.L(2);
    const fpdtype rhow = Q.L(3);
    Q.L(1) =
        rotation(0, 0) * rhou + rotation(0, 1) * rhov + rotation(0, 2) * rhow;
    Q.L(2) =
        rotation(1, 0) * rhou + rotation(1, 1) * rhov + rotation(1, 2) * rhow;
    Q.L(3) =
        rotation(2, 0) * rhou + rotation(2, 1) * rhov + rotation(2, 2) * rhow;
  }
};

struct postDqDxyz {
  haloInOut grads;
  plainIn rotation;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // every gradient turns with the face
    for (int l = 0; l < ne - 1; l++) {
      fpdtype grad[3] = {grads.L(l, 0), grads.L(l, 1), grads.L(l, 2)};
      for (int r = 0; r < 3; r++) {
        fpdtype turned = 0.0;
        for (int c = 0; c < 3; c++) {
          turned += rotation(r, c) * grad[c];
        }
        grads.L(l, r) = turned;
      }
    }
  }
};

} // namespace periodicRot

#endif

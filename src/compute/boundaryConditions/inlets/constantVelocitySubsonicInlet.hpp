#ifndef __constantVelocitySubsonicInlet_H__
#define __constantVelocitySubsonicInlet_H__

#include "kernel.hpp"

namespace constantVelocitySubsonicInlet {

struct euler {
  haloInOut q;
  blockFaceIn qBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // extrapolate pressure, each halo layer mirrored about the first interior
    // cell
    q.L(0) = 2.0 * q.R(0) - q.at(q.p.g + 1, 0);

    // apply velo in halo
    q.L(1) = qBcVals(1);
    q.L(2) = qBcVals(2);
    q.L(3) = qBcVals(3);

    // apply temperature in halo
    q.L(4) = qBcVals(4);

    // apply species in halo
    for (int n = 5; n < ne; n++) {
      q.L(n) = qBcVals(n);
    }
  }
};

struct postDqDxyz {
  haloInOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    for (int l = 0; l < ne; l++) {
      // neumann all gradients
      for (int d = 0; d < 3; d++) {
        grads.L(l, d) = grads.R(l, d);
      }
    }
  }
};

} // namespace constantVelocitySubsonicInlet

#endif

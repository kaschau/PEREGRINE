#ifndef __constantVelocitySubsonicInlet_H__
#define __constantVelocitySubsonicInlet_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"

namespace constantVelocitySubsonicInlet {

struct euler {
  haloInOut Q, q, qh;
  blockFaceIn qBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // the pressure extrapolated to the halo, each layer mirrored about the
    // first interior cell; the face's velocity, temperature and species
    const fpdtype p = 2.0 * q.R(0) - q.at(q.p.g + 1, 0);
    haloState(Q, q, qh, p, qBcVals(4), qBcVals(1), qBcVals(2), qBcVals(3),
              primitiveY(qBcVals));
  }
};

struct postDqDxyz {
  haloInOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // neumann all gradients
    for (int l = 0; l < ne - 1; l++) {
      for (int d = 0; d < 3; d++) {
        grads.L(l, d) = grads.R(l, d);
      }
    }
  }
};

} // namespace constantVelocitySubsonicInlet

#endif

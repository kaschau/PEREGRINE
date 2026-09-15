// Set aside: not a valid condition until the equation of state composes
// into the bc; its post-eos fix-up needs a halo density and does the
// interior cell and the halo in sequence.
#ifndef __constantMassFluxSubsonicInlet_H__
#define __constantMassFluxSubsonicInlet_H__

#include "kernel.hpp"

namespace constantMassFluxSubsonicInlet {

struct euler {
  haloInOut q;
  blockFaceIn qBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // extrapolate pressure, each halo layer mirrored about the first interior
    // cell
    q.L(0) = 2.0 * q.R(0) - q.at(q.p.g + 1, 0);

    // apply zero velo to halo to make subsequent updates easier
    q.L(1) = 0.0;
    q.L(2) = 0.0;
    q.L(3) = 0.0;

    // apply temperature to halo
    q.L(4) = qBcVals(4);

    // apply species to halo
    for (int n = 5; n < ne; n++) {
      q.L(n) = qBcVals(n);
    }
  }
};

struct postEos {
  haloInOut q, Q;
  blockFaceIn QBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // the eos has run on the halo from python: density is valid. The
    // velocities either side of the face are set so that
    // 1/2(rho1+rho2)*1/2(u1+u2) evaluates to the desired mass flux: the
    // first interior layer from the halo layer facing it, then the halo from
    // the interior just set; one thread does both, in that order
    const double &rhou = QBcVals(1);
    const double &rhov = QBcVals(2);
    const double &rhow = QBcVals(3);

    q.R(1) = 4.0 * rhou / (Q.R(0) + Q.L(0)) - q.L(1);
    q.R(2) = 4.0 * rhov / (Q.R(0) + Q.L(0)) - q.L(2);
    q.R(3) = 4.0 * rhow / (Q.R(0) + Q.L(0)) - q.L(3);
    Q.R(1) = q.R(1) * Q.R(0);
    Q.R(2) = q.R(2) * Q.R(0);
    Q.R(3) = q.R(3) * Q.R(0);

    q.L(1) = 4.0 * rhou / (Q.L(0) + Q.R(0)) - q.R(1);
    q.L(2) = 4.0 * rhov / (Q.L(0) + Q.R(0)) - q.R(2);
    q.L(3) = 4.0 * rhow / (Q.L(0) + Q.R(0)) - q.R(3);
    Q.L(1) = q.L(1) * Q.L(0);
    Q.L(2) = q.L(2) * Q.L(0);
    Q.L(3) = q.L(3) * Q.L(0);
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

} // namespace constantMassFluxSubsonicInlet

#endif

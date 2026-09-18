#ifndef __constantMassFluxSubsonicInlet_H__
#define __constantMassFluxSubsonicInlet_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"

namespace constantMassFluxSubsonicInlet {

struct euler {
  haloInOut Q, q, qh;
  blockFaceIn qBcVals, QBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // the pressure extrapolated to the halo, each layer mirrored about the
    // first interior cell; the face's temperature and species; and, with
    // the halo density the eos gives those, the halo velocity that makes the
    // face's average of density times velocity the target momentum
    const fpdtype p = 2.0 * q.R(0) - q.at(q.p.g + 1, 0);
    const auto Y = primitiveY(qBcVals);
    const auto s = eos::fromPrims(p, qBcVals(4), Y, keepHi(haloOf(qh)));
    const auto in = interiorOf(Q);
    const fpdtype scale = 4.0 / (Q.R(0) + s.rho);
    writeState(s, QBcVals(1) * scale - in.u, QBcVals(2) * scale - in.v,
               QBcVals(3) * scale - in.w, Y, haloOf(Q), haloOf(q), haloOf(qh));
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

} // namespace constantMassFluxSubsonicInlet

#endif

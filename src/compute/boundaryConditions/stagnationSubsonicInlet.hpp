#ifndef __stagnationSubsonicInlet_H__
#define __stagnationSubsonicInlet_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"
#include "utils/normal.hpp"

namespace stagnationSubsonicInlet {

struct euler {
  haloInOut Q, q, qh;
  blockFaceIn S, qBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    fpdtype area, nx, ny, nz;
    faceNormal(S(0), S(1), S(2), area, nx, ny, nz);

    // the face's stagnation pressure and temperature, and the interior's
    // Riemann invariant, give the face's velocity and Mach number
    const fpdtype &gamma = qh.R(0);
    const fpdtype &c = qh.R(3);
    const auto in = interiorOf(Q);
    const fpdtype Un = in.u * nx + in.v * ny + in.w * nz;
    const fpdtype V2 = in.u * in.u + in.v * in.v + in.w * in.w;
    const fpdtype Ht = c * c / (gamma - 1.0) + 0.5 * V2;
    const fpdtype Jm = -Un + 2.0 * c / (gamma - 1.0);
    const fpdtype aq = 1 + 2.0 / (gamma - 1.0);
    const fpdtype bq = -2.0 * Jm;
    const fpdtype cq = (gamma - 1.0) * (0.5 * Jm * Jm - Ht);
    const fpdtype t1 = -bq / (2.0 * aq);
    const fpdtype t2 = sqrt(bq * bq - 4.0 * aq * cq) / (2.0 * aq);
    const fpdtype cb = fmax(t1 + t2, t1 - t2);
    const fpdtype Vb = 2.0 * cb / (gamma - 1.0) - Jm;
    const fpdtype Mb = Vb / cb;
    const fpdtype isentropic = 1.0 + (gamma - 1.0) / 2.0 * Mb * Mb;
    const fpdtype p = qBcVals(0) * pow(isentropic, -gamma / (gamma - 1.0));
    const fpdtype T = qBcVals(4) / isentropic;
    haloState(Q, q, qh, p, T, Vb * nx, Vb * ny, Vb * nz, primitiveY(qBcVals));
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

} // namespace stagnationSubsonicInlet

#endif

#ifndef __stagnationSubsonicInlet_H__
#define __stagnationSubsonicInlet_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"

namespace stagnationSubsonicInlet {

struct euler {
  haloInOut Q, q, qh;
  blockFaceIn S, qBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double area, nx, ny, nz;
    faceNormal(S(0), S(1), S(2), area, nx, ny, nz);

    // the face's stagnation pressure and temperature, and the interior's
    // Riemann invariant, give the face's velocity and Mach number
    const double &gamma = qh.R(0);
    const double &c = qh.R(3);
    const auto in = interiorOf(Q);
    const double Un = in.u * nx + in.v * ny + in.w * nz;
    const double V2 = in.u * in.u + in.v * in.v + in.w * in.w;
    const double Ht = c * c / (gamma - 1.0) + 0.5 * V2;
    const double Jm = -Un + 2.0 * c / (gamma - 1.0);
    const double aq = 1 + 2.0 / (gamma - 1.0);
    const double bq = -2.0 * Jm;
    const double cq = (gamma - 1.0) * (0.5 * Jm * Jm - Ht);
    const double t1 = -bq / (2.0 * aq);
    const double t2 = sqrt(bq * bq - 4.0 * aq * cq) / (2.0 * aq);
    const double cb = fmax(t1 + t2, t1 - t2);
    const double Vb = 2.0 * cb / (gamma - 1.0) - Jm;
    const double Mb = Vb / cb;
    const double isentropic = 1.0 + (gamma - 1.0) / 2.0 * Mb * Mb;
    const double p = qBcVals(0) * pow(isentropic, -gamma / (gamma - 1.0));
    const double T = qBcVals(4) / isentropic;
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

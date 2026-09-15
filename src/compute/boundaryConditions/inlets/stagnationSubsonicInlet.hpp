#ifndef __stagnationSubsonicInlet_H__
#define __stagnationSubsonicInlet_H__

#include "kernel.hpp"

namespace stagnationSubsonicInlet {

struct euler {
  faceInOut q;
  faceIn qh, S, qBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double area, nx, ny, nz;
    faceNormal(S.R(0), S.R(1), S.R(2), area, nx, ny, nz);

    // neumann total enthalpy, gamma to halo
    const double &gamma = qh.R(0);
    double uxi = q.R(1) * nx;
    double uvi = q.R(2) * ny;
    double uwi = q.R(3) * nz;
    // Interior velo normal to face
    double Un = uxi + uvi + uwi;

    double V = sqrt(pow(q.R(1), 2.0) + pow(q.R(2), 2.0) + pow(q.R(3), 2.0));
    double Ht = pow(qh.R(3), 2.0) / (gamma - 1.0) + 0.5 * pow(V, 2.0);
    double Jm = -Un + 2.0 * qh.R(3) / (gamma - 1.0);

    // solve quadratic for cb = -b/2a +/- sqrt(b**2-4ac)/2a
    double aq = 1 + 2.0 / (gamma - 1.0);
    double bq = -2.0 * Jm;
    double cq = (gamma - 1.0) * (0.5 * pow(Jm, 2.0) - Ht);
    double t1 = -bq / (2.0 * aq);
    double t2 = sqrt(pow(bq, 2.0) - 4.0 * aq * cq) / (2.0 * aq);

    double cb = fmax(t1 + t2, t1 - t2);

    // boundary velocity, Ma
    double Vb = 2.0 * cb / (gamma - 1.0) - Jm;
    double Mb = Vb / cb;

    // compute static pressure
    q.L(0) = qBcVals.here(0) * pow(1.0 + (gamma - 1.0) / 2.0 * pow(Mb, 2.0),
                                   -gamma / (gamma - 1.0));

    // extrapolate face normal velocity
    q.L(1) = Vb * nx;
    q.L(2) = Vb * ny;
    q.L(3) = Vb * nz;

    // compute static temperature
    q.L(4) = qBcVals.here(4) / (1.0 + (gamma - 1.0) / 2.0 * pow(Mb, 2.0));

    // apply species in halo
    for (int n = 5; n < ne; n++) {
      q.L(n) = qBcVals.here(n);
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

} // namespace stagnationSubsonicInlet

#endif

#ifndef __supersonicExit_H__
#define __supersonicExit_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"

namespace supersonicExit {

struct euler {
  haloInOut Q, q, qh;
  blockFaceIn S;
  static constexpr double floor = 0.01;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const double dplus = S.outward();
    double area, nx, ny, nz;
    faceNormal(S(0), S(1), S(2), area, nx, ny, nz);

    // everything extrapolated to the halo, each layer mirrored about the
    // first interior cell: the pressure no higher than the interior's, the
    // velocity flipped on the face instead where reverse flow is detected,
    // the mass fractions kept in [0, 1]; p and T no lower than a hundredth
    // of the interior's: a safety net for a transient reaching the exit,
    // never binding on a smooth outflow, so the halo keeps a density
    const double p =
        fmin(fmax(floor * q.R(0), 2.0 * q.R(0) - q.at(q.p.g + 1, 0)), q.R(0));
    const auto in = interiorOf(Q);
    const double rhoinv2 = 1.0 / Q.at(Q.p.g + 1, 0);
    const double uDotn = (in.u * nx + in.v * ny + in.w * nz) * dplus;
    double u, v, w;
    if (uDotn > 0.0) {
      u = 2.0 * in.u - Q.at(Q.p.g + 1, 1) * rhoinv2;
      v = 2.0 * in.v - Q.at(Q.p.g + 1, 2) * rhoinv2;
      w = 2.0 * in.w - Q.at(Q.p.g + 1, 3) * rhoinv2;
    } else {
      u = in.u - 2.0 * uDotn * nx * dplus;
      v = in.v - 2.0 * uDotn * ny * dplus;
      w = in.w - 2.0 * uDotn * nz * dplus;
    }
    const double T = fmax(floor * q.R(1), 2.0 * q.R(1) - q.at(q.p.g + 1, 1));
    const auto Y = [&](const int n) {
      if (n < ns - 1) {
        return fmax(0.0, fmin(1.0, 2.0 * Q.R(5 + n) * in.rhoinv -
                                       Q.at(Q.p.g + 1, 5 + n) * rhoinv2));
      }
      double last = 1.0;
      for (int m = 0; m < ns - 1; m++) {
        last -= fmax(0.0, fmin(1.0, 2.0 * Q.R(5 + m) * in.rhoinv -
                                        Q.at(Q.p.g + 1, 5 + m) * rhoinv2));
      }
      return last;
    };
    haloState(Q, q, qh, p, T, u, v, w, Y);
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

} // namespace supersonicExit

#endif

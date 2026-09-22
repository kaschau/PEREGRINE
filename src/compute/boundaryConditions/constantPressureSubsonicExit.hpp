#ifndef __constantPressureSubsonicExit_H__
#define __constantPressureSubsonicExit_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"
#include "utils/normal.hpp"

namespace constantPressureSubsonicExit {

struct euler {
  haloInOut Q, q, qh;
  blockFaceIn S, qBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype dplus = S.outward();

    fpdtype area, nx, ny, nz;
    faceNormal(S(0), S(1), S(2), area, nx, ny, nz);

    // the velocity extrapolated to the halo, each layer mirrored about the
    // first interior cell, unless reverse flow is detected, then flipped on
    // the face like a slip wall; the face's pressure; everything else neumann
    const auto in = interiorOf(Q);
    const fpdtype uDotn = (in.u * nx + in.v * ny + in.w * nz) * dplus;
    fpdtype u, v, w;
    if (uDotn > 0.0) {
      const fpdtype rhoinv2 = 1.0 / Q.at(Q.p.g + 1, 0);
      u = 2.0 * in.u - Q.at(Q.p.g + 1, 1) * rhoinv2;
      v = 2.0 * in.v - Q.at(Q.p.g + 1, 2) * rhoinv2;
      w = 2.0 * in.w - Q.at(Q.p.g + 1, 3) * rhoinv2;
    } else {
      u = in.u - 2.0 * uDotn * nx * dplus;
      v = in.v - 2.0 * uDotn * ny * dplus;
      w = in.w - 2.0 * uDotn * nz * dplus;
    }
    haloState(Q, q, qh, qBcVals(0), q.R(1), u, v, w,
              massFractions(cellOf(Q, 0), in.rhoinv));
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

} // namespace constantPressureSubsonicExit

#endif

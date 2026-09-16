#ifndef __adiabaticSlipWall_H__
#define __adiabaticSlipWall_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"

namespace adiabaticSlipWall {

struct euler {
  haloInOut Q, q, qh;
  blockFaceIn S;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double area, nx, ny, nz;
    faceNormal(S(0), S(1), S(2), area, nx, ny, nz);

    // mirror the velocity about the wall; pressure and species match
    const auto in = interiorOf(Q);
    const double uDotn = in.u * nx + in.v * ny + in.w * nz;
    haloState(Q, q, qh, q.R(0), q.R(1), in.u - 2.0 * uDotn * nx,
              in.v - 2.0 * uDotn * ny, in.w - 2.0 * uDotn * nz,
              massFractions(cellOf(Q, 0), in.rhoinv));
  }
};

struct postDqDxyz {
  haloInOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    for (int d = 0; d < 3; d++) {
      // velocity gradients negated, so the wall gradient is zero
      for (int l = 0; l < 3; l++) {
        grads.L(l, d) = -grads.R(l, d);
      }
      // temperature gradient negated, so the wall gradient is zero
      grads.L(3, d) = -grads.R(3, d);
      // species gradients negated, so the wall gradient is zero
      for (int l = 4; l < ne - 1; l++) {
        grads.L(l, d) = -grads.R(l, d);
      }
    }
  }
};

} // namespace adiabaticSlipWall

#endif

// Hendrickson, Kartha and Candler (AIAA 2018-3710): the Ducros ratio,
// dilatation squared over dilatation plus vorticity squared, sharpened as
// US3D does and confined to shocks by a high-pass on the dilatation at
// nu |u| / l, l the cell's length; floored, at the cell either side of the
// face, the larger. The velocity gradients are the right-hand side's,
// grads, so the switch is the viscous flow's. nu and the floor are the
// config's switchValues, baked in.
#ifndef __switchDucros_H__
#define __switchDucros_H__

#include "faces.hpp"

#if !defined(PG_DUCROS_NU) || !defined(PG_DUCROS_FLOOR)
#error "ducros takes switchValues nu and floor"
#endif

struct ducros : PG_RECONSTRUCT {
  using base = PG_RECONSTRUCT;
  cellStradMatIn grads;
  cellStradScalIn cellLength;
  template <class P> KOKKOS_INLINE_FUNCTION void pin(const P &at) {
    pinKernel(static_cast<base &>(*this), at);
    pinEach(at, grads, cellLength);
  }

  template <class G, class Q>
  static KOKKOS_INLINE_FUNCTION fpdtype sensor(const G &grads, const Q &Q_,
                                               const fpdtype cellLength) {
    // the velocity gradient rows: u, v, w by x, y, z
    const fpdtype div = grads(0, 0) + grads(1, 1) + grads(2, 2);
    const fpdtype div2 = div * div;
    const fpdtype wx = grads(2, 1) - grads(1, 2),
                  wy = grads(0, 2) - grads(2, 0),
                  wz = grads(1, 0) - grads(0, 1);
    const fpdtype omega2 = wx * wx + wy * wy + wz * wz;
    // quiescent flow (0 / 0) guards to 0
    const fpdtype tiny = 1e-300;
    const fpdtype D = fmin(4.0 / 3.0 * div2 / (div2 + omega2 + tiny), 1.0);
    const fpdtype rhoinv = 1.0 / Q_(0);
    const fpdtype u2 =
        (Q_(1) * Q_(1) + Q_(2) * Q_(2) + Q_(3) * Q_(3)) * rhoinv * rhoinv;
    const fpdtype l = cellLength;
    const fpdtype cutoff2 = PG_DUCROS_NU * PG_DUCROS_NU * u2 / (l * l);
    const fpdtype theta = div2 / (div2 + cutoff2 + tiny);
    return fmax(D - PG_DUCROS_FLOOR, 0.0) * theta + PG_DUCROS_FLOOR;
  }
  KOKKOS_INLINE_FUNCTION fpdtype weight() const {
    return fmax(sensor(grads.L(), this->Q.L(), cellLength.L()),
                sensor(grads.R(), this->Q.R(), cellLength.R()));
  }
};

#endif

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
  cellCenterL gradsL, cellLengthL;
  cellCenterR gradsR, cellLengthR;
  template <class P> KOKKOS_INLINE_FUNCTION void pin(const P &at) {
    pinAll(static_cast<base &>(*this), at);
    pinEach(at, gradsL, cellLengthL, gradsR, cellLengthR);
  }

  template <class G, class Q, class L>
  static KOKKOS_INLINE_FUNCTION double sensor(const G &grads, const Q &Q_,
                                              const L &cellLength) {
    // the velocity gradient rows: u, v, w by x, y, z
    const double div = grads(0, 0) + grads(1, 1) + grads(2, 2);
    const double div2 = div * div;
    const double wx = grads(2, 1) - grads(1, 2), wy = grads(0, 2) - grads(2, 0),
                 wz = grads(1, 0) - grads(0, 1);
    const double omega2 = wx * wx + wy * wy + wz * wz;
    // quiescent flow (0 / 0) guards to 0
    const double tiny = 1e-300;
    const double D = fmin(4.0 / 3.0 * div2 / (div2 + omega2 + tiny), 1.0);
    const double rhoinv = 1.0 / Q_(0);
    const double u2 =
        (Q_(1) * Q_(1) + Q_(2) * Q_(2) + Q_(3) * Q_(3)) * rhoinv * rhoinv;
    const double l = cellLength();
    const double cutoff2 = PG_DUCROS_NU * PG_DUCROS_NU * u2 / (l * l);
    const double theta = div2 / (div2 + cutoff2 + tiny);
    return fmax(D - PG_DUCROS_FLOOR, 0.0) * theta + PG_DUCROS_FLOOR;
  }
  KOKKOS_INLINE_FUNCTION double weight() const {
    return fmax(sensor(gradsL, this->QL, cellLengthL),
                sensor(gradsR, this->QR, cellLengthR));
  }
};

#endif

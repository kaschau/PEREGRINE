// Piecewise constant: the face takes each side's cell as it is, first
// order.
// A reconstruction is the flux kernel: its columns, the two states it
// makes of them, a species' partial density either side, and the body
// that hands them to the Riemann solver, PG_RIEMANN, forced in ahead.
#ifndef __reconstructPiecewiseConstant_H__
#define __reconstructPiecewiseConstant_H__

#include "advFlux/faceState.hpp"
#include "faces.hpp"

struct piecewiseConstant {
  cellCenterL QL, qL, qhL;
  cellCenterR QR, qR, qhR;
  cellFaceOut F;
  cellFaceIn A;

  template <class C>
  static KOKKOS_INLINE_FUNCTION faceState state(const C &Q, const C &q,
                                                const C &qh) {
    const double rhoinv = 1.0 / Q(0);
    return {Q(0), Q(1) * rhoinv, Q(2) * rhoinv, Q(3) * rhoinv, Q(1),
            Q(2), Q(3),          q(0),          Q(4),          qh(3)};
  }
  struct sides {
    faceState L, R;
  };
  KOKKOS_INLINE_FUNCTION sides states() const {
    return {state(QL, qL, qhL), state(QR, qR, qhR)};
  }
  KOKKOS_INLINE_FUNCTION void species(const sides &, int n, double &L,
                                      double &R) const {
    L = QL(5 + n);
    R = QR(5 + n);
  }
  KOKKOS_INLINE_FUNCTION void operator()() const {
    PG_RIEMANN::flux(*this, A, F);
  }
};

#endif

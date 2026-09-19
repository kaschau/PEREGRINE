// Piecewise constant: the face takes each side's cell as it is, first
// order.
// A reconstruction is the flux kernel's base: its columns, the two states
// it makes of them, and a species' partial density either side, which the
// flux formula takes.
#ifndef __reconstructPiecewiseConstant_H__
#define __reconstructPiecewiseConstant_H__

#include "advFlux/faceState.hpp"
#include "faces.hpp"

struct piecewiseConstant {
  cellStradVecIn Q, q, qh;
  faceVecOut F;
  faceVecIn A;

  template <class C>
  static KOKKOS_INLINE_FUNCTION faceState state(const C &Q, const C &q,
                                                const C &qh) {
    const fpdtype rhoinv = 1.0 / Q(0);
    return {Q(0), Q(1) * rhoinv, Q(2) * rhoinv, Q(3) * rhoinv, Q(1),
            Q(2), Q(3),          q(0),          Q(4),          qh(3)};
  }
  struct sides {
    faceState L, R;
  };
  KOKKOS_INLINE_FUNCTION sides states() const {
    return {state(Q.L(), q.L(), qh.L()), state(Q.R(), q.R(), qh.R())};
  }
  KOKKOS_INLINE_FUNCTION void species(const sides &, int n, fpdtype &L,
                                      fpdtype &R) const {
    L = Q.L(5 + n);
    R = Q.R(5 + n);
  }
};

#endif

#include "eos.hpp"
#include "kernel.hpp"

// The state of every cell from its conserved variables: the mass fractions
// clipped and renormalized in Q, then the case's eos for p, T and qh; T's
// last value is the iterative eos's guess.
PG_RANGE(cellCenters)
struct stateFromCons {
  cellVecInOut Q, q;
  cellVecOut qh;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype rho = Q(0);
    const fpdtype rhoinv = 1.0 / rho;
    const fpdtype tke =
        0.5 * (Q(1) * Q(1) + Q(2) * Q(2) + Q(3) * Q(3)) * rhoinv;

    // the species mass clipped to [0, rho], and scaled down where the mass
    // fractions sum past one
    fpdtype sum = 0.0;
    for (int n = 0; n < ns - 1; n++) {
      Q(5 + n) = fmax(fmin(Q(5 + n), rho), 0.0);
      sum += Q(5 + n);
    }
    if (sum > rho) {
      const fpdtype scale = rho / sum;
      for (int n = 0; n < ns - 1; n++) {
        Q(5 + n) *= scale;
      }
    }

    const fpdtype e = (Q(4) - tke) * rhoinv;
    const auto s =
        eos::fromCons(rho, e, massFractions(Q, rhoinv), q(1), keepHi(qh));
    q(0) = s.p;
    q(1) = s.T;
    qh(0) = s.gamma;
    qh(1) = s.cp;
    qh(2) = rho * s.h;
    qh(3) = s.c;
    qh(4) = rho * e;
  }
};

PG_ABI void pgStateFromCons(const stateFromCons &k, const pgTiling &t) {
  forCells("state from conserved", t, k);
}

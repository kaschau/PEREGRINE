// The case's equation of state, as a piece any kernel or boundary condition
// can call: eos::fromPrims and eos::fromCons work out the mixture state at
// one point from (p, T, Y) or (rho, e, Y) over the species tables, and hand
// back a `state`; eos::enthalpies gives each species' enthalpy at a cell;
// eos::densityDerivatives what the dual time preconditioning linearizes
// with; eos::qhComponents says how wide a cell's qh is. The jit picks the
// case's eos: it forces the named header (thermo/cpg.hpp, tpg.hpp,
// realGas.hpp) in ahead of the source and defines PG_EOS, so nothing here
// names one. Each eos header declares the same pieces and a `state`; a
// kernel includes this and writes eos::.
#ifndef __eos_H__
#define __eos_H__

#include "conserved.hpp"
#include "species.hpp"

#ifndef PG_EOS
#error                                                                         \
    "an eos kernel is compiled for one equation of state: -DPG_EOS from the jit"
#endif

namespace eos = PG_EOS;

// where fromPrims and fromCons put a species enthalpy: into the cell's
// qh(5 + n) for an eos that keeps them, nowhere for one that does not
template <class Qh> KOKKOS_INLINE_FUNCTION auto keepHi(const Qh &qh) {
  return [qh](const int n, const double h) {
    if constexpr (eos::qhComponents > 5) {
      qh(5 + n) = h;
    }
  };
}

// a cell's state written out of what the eos worked out, given its velocity:
// Q's density, momentum, total energy and species mass; q's p and T; qh's
// gamma, cp, rho h, c, rho e
template <class Yf, class QC, class QP, class QH>
KOKKOS_INLINE_FUNCTION void
writeState(const eos::state &s, const double u, const double v, const double w,
           const Yf &Y, const QC &Q, const QP &q, const QH &qh) {
  Q(0) = s.rho;
  Q(1) = s.rho * u;
  Q(2) = s.rho * v;
  Q(3) = s.rho * w;
  Q(4) = s.rho * (s.e + 0.5 * (u * u + v * v + w * w));
  for (int n = 0; n < ns - 1; n++) {
    Q(5 + n) = s.rho * Y(n);
  }
  q(0) = s.p;
  q(1) = s.T;
  qh(0) = s.gamma;
  qh(1) = s.cp;
  qh(2) = s.rho * s.h;
  qh(3) = s.c;
  qh(4) = s.rho * s.e;
}

#endif

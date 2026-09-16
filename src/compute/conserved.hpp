// What a cell's conserved state gives back without the eos: the velocity by
// one reciprocal of the density, the mass fractions, the last from the rest,
// and from those the mole fractions. A kernel reads these off Q rather than
// keeping them in an array.
#ifndef __conserved_H__
#define __conserved_H__

#include "kernel.hpp"
#include "species.hpp"

// the mass fractions of the cell a column stands on, as a callable of the
// species: Q(5 + n) rhoinv for the first ns - 1, one minus their sum for the
// last
template <class Q>
KOKKOS_INLINE_FUNCTION auto massFractions(const Q &Qc, const double rhoinv) {
  return [=](const int n) {
    if (n < ns - 1) {
      return Qc(5 + n) * rhoinv;
    }
    double last = 1.0;
    for (int m = 0; m < ns - 1; m++) {
      last -= Qc(5 + m) * rhoinv;
    }
    return last;
  };
}

// the same, into an array a kernel loops over more than once
template <class Q>
KOKKOS_INLINE_FUNCTION void massFractions(const Q &Qc, const double rhoinv,
                                          double *Y) {
  Y[ns - 1] = 1.0;
  for (int n = 0; n < ns - 1; n++) {
    Y[n] = Qc(5 + n) * rhoinv;
    Y[ns - 1] -= Y[n];
  }
}

// the mole fractions from the mass fractions Y(n), and the mean molecular
// weight: Y sums to one, so sum(X MW) is 1 / sum(Y / MW)
template <class Yf>
KOKKOS_INLINE_FUNCTION double moleFractions(const Yf &Y, double *X) {
  double denom = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    denom += Y(n) * MWinv(n);
  }
  const double MWmix = 1.0 / denom;
  for (int n = 0; n <= ns - 1; n++) {
    X[n] = Y(n) * MWinv(n) * MWmix;
  }
  return MWmix;
}

#endif

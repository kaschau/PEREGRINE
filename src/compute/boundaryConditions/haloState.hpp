// What an euler bcHook starts from and ends with. A hook forms the halo
// cell's p, T, velocity and mass fractions -- from the interior cell's, read
// off its conserved state, and from the face's values -- then hands them to
// haloState, which writes the halo cell's conserved and thermodynamic state
// through the case's eos. A hook leaves a consistent halo; nothing runs
// after it.
#ifndef __haloState_H__
#define __haloState_H__

#include "kernel.hpp"
#include "thermo/eos.hpp"

// the cell `layer` from the face of a halo column, as a callable of the
// component: 0 the first interior cell, negative into the halo
template <class H>
KOKKOS_INLINE_FUNCTION auto cellOf(const H &column, const int layer) {
  return [=](const int c) -> decltype(auto) { return column.at(layer, c); };
}

// the halo cell a halo column stands on, as a callable of the component
template <class H> KOKKOS_INLINE_FUNCTION auto haloOf(const H &column) {
  return [=](const int c) -> decltype(auto) { return column.L(c); };
}

// the first interior cell of a halo column, read off its conserved state
struct interiorCell {
  fpdtype rhoinv, u, v, w;
};
template <class H> KOKKOS_INLINE_FUNCTION interiorCell interiorOf(const H &Q) {
  const fpdtype rhoinv = 1.0 / Q.R(0);
  return {rhoinv, Q.R(1) * rhoinv, Q.R(2) * rhoinv, Q.R(3) * rhoinv};
}

// the mass fractions of a primitive vector p, u, v, w, T, Y(0 .. ns - 2), as
// a callable of the species, the last from the rest
template <class V> KOKKOS_INLINE_FUNCTION auto primitiveY(const V &vals) {
  return [=](const int n) {
    if (n < ns - 1) {
      return vals(5 + n);
    }
    fpdtype last = 1.0;
    for (int m = 0; m < ns - 1; m++) {
      last -= vals(5 + m);
    }
    return last;
  };
}

// the halo cell's state from its primitives, through the eos: Q's density,
// momentum, total energy and species mass, q's p and T, qh's mixture
// properties and, for an eos that keeps them, species enthalpies
template <class HQ, class Hq, class Hqh, class Yf>
KOKKOS_INLINE_FUNCTION void haloState(const HQ &Q, const Hq &q, const Hqh &qh,
                                      const fpdtype p, const fpdtype T,
                                      const fpdtype u, const fpdtype v,
                                      const fpdtype w, const Yf &Y) {
  const auto s = eos::fromPrims(p, T, Y, keepHi(haloOf(qh)));
  writeState(s, u, v, w, Y, haloOf(Q), haloOf(q), haloOf(qh));
}

#endif

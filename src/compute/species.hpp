// The species data of the case, baked by the jit: a value per species, or a
// polynomial in ln T per species (ascending coefficients, every fit its own
// degree, nothing padded), reached through one accessor each. The jit
// forces the tables in ahead of any source that includes this; a table is
// a static constexpr array inside a plain inline function, the one form
// both device compilers place well (a namespace-scope constexpr array is a
// host object to nvcc, and a non-static local one is rebuilt on the stack
// per call). Included by the kernels that read species data, and only them.
#ifndef __species_H__
#define __species_H__

#include "arrays.hpp"

#ifndef PG_SPECIES_TABLES
#error                                                                         \
    "a species kernel is compiled with the case's tables, forced in by the jit"
#endif

constexpr double Ru = PG_RU;

// one value per species
#define PG_SPECIES_SCALAR(name, DATA)                                          \
  KOKKOS_INLINE_FUNCTION double name(const int n) {                            \
    static constexpr double t[ns] = DATA;                                      \
    return t[n];                                                               \
  }

// one polynomial per row: its degree, and coefficient m of row r
#define PG_SPECIES_POLYNOMIAL(name, OFFSETS, COEFS)                            \
  KOKKOS_INLINE_FUNCTION int name##Degree(const int r) {                       \
    static constexpr int o[] = OFFSETS;                                        \
    return o[r + 1] - o[r];                                                    \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION double name(const int r, const int m) {               \
    static constexpr int o[] = OFFSETS;                                        \
    static constexpr double c[] = COEFS;                                       \
    return c[o[r] + m];                                                        \
  }

PG_SPECIES_SCALAR(MW, PG_MW)
PG_SPECIES_SCALAR(hRef, PG_H_REF)
PG_SPECIES_SCALAR(cp0, PG_CP0)
PG_SPECIES_SCALAR(mu0, PG_MU0)
PG_SPECIES_SCALAR(kappa0, PG_KAPPA0)
PG_SPECIES_SCALAR(lewis, PG_LEWIS)
PG_SPECIES_SCALAR(Tcrit, PG_TCRIT)
PG_SPECIES_SCALAR(pcrit, PG_PCRIT)
PG_SPECIES_SCALAR(Vcrit, PG_VCRIT)
PG_SPECIES_SCALAR(acentric, PG_ACENTRIC)
PG_SPECIES_SCALAR(redDipole, PG_RED_DIPOLE)

PG_SPECIES_POLYNOMIAL(cpPoly, PG_CP_POLY_OFFSETS, PG_CP_POLY_COEFS)
PG_SPECIES_POLYNOMIAL(hPoly, PG_H_POLY_OFFSETS, PG_H_POLY_COEFS)
PG_SPECIES_POLYNOMIAL(sPoly, PG_S_POLY_OFFSETS, PG_S_POLY_COEFS)
PG_SPECIES_POLYNOMIAL(muPoly, PG_MU_POLY_OFFSETS, PG_MU_POLY_COEFS)
PG_SPECIES_POLYNOMIAL(kappaPoly, PG_KAPPA_POLY_OFFSETS, PG_KAPPA_POLY_COEFS)
PG_SPECIES_POLYNOMIAL(chungA, PG_CHUNG_A_OFFSETS, PG_CHUNG_A_COEFS)
PG_SPECIES_POLYNOMIAL(chungB, PG_CHUNG_B_OFFSETS, PG_CHUNG_B_COEFS)

// the binary diffusion fits are one per unordered pair, the rows the pairs
// i <= j in numpy's triu_indices order: row i's pairs start after the
// i rows before it, which held ns, ns - 1, ... pairs
PG_SPECIES_POLYNOMIAL(dijPair, PG_DIJ_OFFSETS, PG_DIJ_COEFS)
KOKKOS_INLINE_FUNCTION int pairIndex(const int n, const int n2) {
  const int i = n < n2 ? n : n2, j = n < n2 ? n2 : n;
  return i * ns - i * (i - 1) / 2 + (j - i);
}
KOKKOS_INLINE_FUNCTION int dijDegree(const int n, const int n2) {
  return dijPairDegree(pairIndex(n, n2));
}
KOKKOS_INLINE_FUNCTION double dij(const int n, const int n2, const int m) {
  return dijPair(pairIndex(n, n2), m);
}

#endif
